import os
import cv2
import time
import argparse
import numpy as np
import torch
import gc
import threading
import glob
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from ultralytics import YOLO
from sqlalchemy.orm import scoped_session

# Importaciones de tu proyecto
from src.database.connection import SessionLocal
from src.database.models import VideoMetadata, FrameObjectDetection, GroupDetection, GroupMember

# DeepSORT Check
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    SORT_AVAILABLE = True
except ImportError:
    print("ERROR: Deep SORT no disponible. Instala: pip install deep-sort-realtime")
    exit(1)

# Lock global para impresiones en consola ordenadas
print_lock = threading.Lock()
# Lock global para creación de videos en BD (evitar duplicados en race conditions)
db_lock = threading.Lock()

def safe_print(message):
    """Imprime mensajes de forma segura entre hilos"""
    with print_lock:
        print(message)

# ==========================================
# CLASE 1: LOGICA DE GRUPOS (ALGORITMO PDF)
# ==========================================
class GroupTracker:
    def __init__(self, distance_threshold=100, min_frames=15):
        self.distance_threshold = distance_threshold
        self.min_frames = min_frames
        self.next_group_id = 1
        self.pair_frames = defaultdict(int)
        self.group_assignments = {}
        self.group_history = defaultdict(lambda: defaultdict(int))

    def calculate_distance(self, pos1, pos2):
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def calculate_centroid(self, box):
        return ((box['x1'] + box['x2']) / 2, (box['y1'] + box['y2']) / 2)

    def update(self, detections):
        current_centroids = {det['track_id']: self.calculate_centroid(det) for det in detections}
        track_ids = list(current_centroids.keys())
        current_pairs = set()

        # Detección de pares
        for i in range(len(track_ids)):
            for j in range(i + 1, len(track_ids)):
                id1, id2 = track_ids[i], track_ids[j]
                dist = self.calculate_distance(current_centroids[id1], current_centroids[id2])
                if dist < self.distance_threshold:
                    current_pairs.add(tuple(sorted([id1, id2])))

        # Persistencia temporal
        for pair in current_pairs:
            self.pair_frames[pair] += 1
        for pair in set(self.pair_frames.keys()) - current_pairs:
            self.pair_frames[pair] = 0

        confirmed_pairs = {pair for pair, frames in self.pair_frames.items() if frames >= self.min_frames}

        # Construcción de grupos (Grafos)
        graph = defaultdict(set)
        for id1, id2 in confirmed_pairs:
            graph[id1].add(id2)
            graph[id2].add(id1)

        visited = set()
        current_groups_list = []
        for node in graph:
            if node not in visited:
                stack = [node]
                component = set()
                while stack:
                    curr = stack.pop()
                    if curr not in visited:
                        visited.add(curr)
                        component.add(curr)
                        stack.extend(graph[curr] - visited)
                current_groups_list.append(component)

        # Asignación de IDs
        new_assignments = {}
        for members in current_groups_list:
            group_id = None
            for m in members:
                if m in self.group_assignments:
                    group_id = self.group_assignments[m]
                    break
            
            if group_id is None:
                group_id = self.next_group_id
                self.next_group_id += 1
            
            for m in members:
                new_assignments[m] = group_id
                self.group_history[group_id][m] += 1

        self.group_assignments = new_assignments
        
        output = []
        unique_groups = defaultdict(list)
        for tid, gid in self.group_assignments.items():
            unique_groups[gid].append(tid)
            
        for gid, members in unique_groups.items():
            if len(members) >= 2:
                output.append({'group_id': gid, 'members': members, 'size': len(members)})
        return output

# ==========================================
# FUNCION WORKER (PROCESA UN SOLO VIDEO)
# ==========================================
def process_single_video(video_path, output_stats_dir, model_path, config):
    """
    Función que se ejecutará en cada hilo.
    """
    video_name = os.path.basename(video_path)
    safe_print(f"--> Iniciando hilo para: {video_name}")
    
    # 1. Configuración Local del Hilo (DB y Modelo)
    # IMPORTANTE: Crear una sesión de DB nueva para este hilo
    db = SessionLocal()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Cargar modelo (Ultralytics es thread-safe generalmente, pero instanciamos por seguridad)
    try:
        model = YOLO(model_path)
        model.to(device)
    except Exception as e:
        safe_print(f"Error cargando modelo para {video_name}: {e}")
        return

    # 2. Registrar Video en BD
    video_record = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            safe_print(f"Error abriendo video: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps else 0
        size_mb = os.path.getsize(video_path) / (1024 * 1024)

        # Usar lock para evitar crear el mismo video dos veces si hay error de concurrencia
        with db_lock:
            video_record = db.query(VideoMetadata).filter_by(title=video_name).first()
            if not video_record:
                video_record = VideoMetadata(title=video_name, duration=duration, size=size_mb)
                db.add(video_record)
                db.commit()
                safe_print(f"    [BD] Nuevo registro creado: {video_name}")
            else:
                safe_print(f"    [BD] Video ya existente: {video_name}")
                # Opcional: Limpiar datos previos si se re-analiza
                # db.query(FrameObjectDetection).filter_by(video_id=video_record.video_id).delete()
                # db.commit()

    except Exception as e:
        safe_print(f"Error inicializando DB para {video_name}: {e}")
        db.close()
        return

    # 3. Inicializar Trackers
    tracker = DeepSort(
        max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.2,
        nn_budget=100, embedder="mobilenet", half=True, bgr=True,
        embedder_gpu=(device == 'cuda')
    )
    
    group_tracker = GroupTracker(
        distance_threshold=config['group_dist'], 
        min_frames=config['min_frames']
    )

    # 4. Bucle de Procesamiento
    frame_idx = 0
    detections_batch = []
    stats = {"groups": 0}
    BATCH_SIZE = 50  # Commit cada 50 frames

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # --- DETECCION ---
            results = model.predict(frame, conf=config['conf'], classes=[0], verbose=False, device=device)
            
            yolo_dets = []
            if results[0].boxes:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf_val = box.conf[0].cpu().item()
                    w, h = x2 - x1, y2 - y1
                    yolo_dets.append(([int(x1), int(y1), int(w), int(h)], conf_val, "person"))

            # --- RASTREO ---
            tracks = tracker.update_tracks(yolo_dets, frame=frame)
            current_frame_tracks = [] 

            for track in tracks:
                if not track.is_confirmed(): continue
                
                track_id = track.track_id
                x1, y1, x2, y2 = track.to_ltrb()

                # Añadir a batch de guardado
                detections_batch.append(FrameObjectDetection(
                    video_id=video_record.video_id, frame_number=frame_idx, track_id=track_id,
                    x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)
                ))

                current_frame_tracks.append({
                    'track_id': track_id, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                })

            # --- GRUPOS ---
            if current_frame_tracks:
                groups_found = group_tracker.update(current_frame_tracks)
                stats["groups"] = len(groups_found)
                
                for grp in groups_found:
                    # Calcular centroide y dispersión
                    centers = []
                    for tid in grp['members']:
                        t = next((t for t in current_frame_tracks if t['track_id'] == tid), None)
                        if t: centers.append([(t['x1']+t['x2'])/2, (t['y1']+t['y2'])/2])
                    
                    cx, cy, disp = 0, 0, 0
                    if centers:
                        centers = np.array(centers)
                        cx, cy = np.mean(centers, axis=0)
                        disp = float(np.mean(np.linalg.norm(centers - np.array([cx, cy]), axis=1)))

                    # Guardar Grupo
                    group_obj = GroupDetection(
                        video_id=video_record.video_id, frame_number=frame_idx,
                        group_id=grp['group_id'], center_x=float(cx), center_y=float(cy),
                        size=grp['size'], dispersion=disp, avg_velocity=0.0, velocity_std=0.0
                    )
                    db.add(group_obj)
                    db.flush() # Obtener ID
                    
                    for mid in grp['members']:
                        db.add(GroupMember(group_detection_id=group_obj.id, track_id=mid))

            # --- BATCH SAVE ---
            if len(detections_batch) >= BATCH_SIZE:
                db.bulk_save_objects(detections_batch)
                db.commit()
                detections_batch = []

            # Log discreto
            if frame_idx % 100 == 0:
                safe_print(f"   [{video_name[:10]}...] Frame {frame_idx} | Grupos activos: {stats['groups']}")

        # Commit final
        if detections_batch:
            db.bulk_save_objects(detections_batch)
        db.commit()

        # 5. Generar Reporte
        generate_report(db, video_record, output_stats_dir)

    except Exception as e:
        safe_print(f"ERROR FATAL en {video_name}: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        db.close()
        # Liberar memoria de modelo y tracker si es posible
        del model
        del tracker
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
        safe_print(f"--> Finalizado: {video_name}")

def generate_report(db, video, output_folder):
    """Genera reporte de texto usando la sesión de DB proporcionada"""
    try:
        os.makedirs(output_folder, exist_ok=True)
        report_path = os.path.join(output_folder, f"REPORTE_{video.title}.txt")
        
        # Estadísticas
        total_frames = db.query(FrameObjectDetection.frame_number)\
            .filter_by(video_id=video.video_id)\
            .order_by(FrameObjectDetection.frame_number.desc()).first()
        last_frame = total_frames[0] if total_frames else 0

        unique_people = db.query(FrameObjectDetection.track_id)\
            .filter_by(video_id=video.video_id).distinct().count()

        all_groups = db.query(GroupDetection).filter_by(video_id=video.video_id).all()
        unique_groups_ids = set(g.group_id for g in all_groups)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"REPORTE AUTOMÁTICO - {video.title}\n")
            f.write(f"{'='*40}\n")
            f.write(f"Duración frames: {last_frame}\n")
            f.write(f"Personas únicas: {unique_people}\n")
            f.write(f"Grupos detectados: {len(unique_groups_ids)}\n")
            
            # Detalle grupos
            group_durations = defaultdict(int)
            for g in all_groups: group_durations[g.group_id] += 1
            
            top_groups = sorted(group_durations.items(), key=lambda x: x[1], reverse=True)[:5]
            if top_groups:
                f.write(f"\nTop 5 Grupos más duraderos:\n")
                for gid, frames in top_groups:
                    f.write(f" - Grupo {gid}: {frames} frames\n")

        safe_print(f"    [REPORT] Generado en: {report_path}")

    except Exception as e:
        safe_print(f"Error generando reporte para {video.title}: {e}")

# ==========================================
# MAIN
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Pipeline Concurrente de Video Analítica")
    parser.add_argument("--input_dir", type=str, required=True, help="Carpeta con videos .mp4")
    parser.add_argument("--output_dir", type=str, default="reportes", help="Carpeta para reportes")
    parser.add_argument("--max_workers", type=int, default=2, help="Número de videos simultáneos (Cuidado con VRAM)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confianza YOLO")
    parser.add_argument("--group_dist", type=float, default=100.0)
    parser.add_argument("--min_frames", type=int, default=15)
    
    args = parser.parse_args()

    # Buscar videos
    video_files = glob.glob(os.path.join(args.input_dir, "*.[mM][pP]4"))
    if not video_files:
        print(f"No se encontraron videos MP4 en {args.input_dir}")
        return

    print(f"\n{'='*60}")
    print(f" INICIANDO PROCESAMIENTO CONCURRENTE")
    print(f" Videos encontrados: {len(video_files)}")
    print(f" Hilos simultáneos: {args.max_workers}")
    print(f" GPU Disponible: {torch.cuda.is_available()}")
    print(f"{'='*60}\n")

    model_path = "yolo11x.pt" # O busca dinámicamente como en tus otros scripts
    
    config = {
        'conf': args.conf,
        'group_dist': args.group_dist,
        'min_frames': args.min_frames
    }

    start_time = time.time()

    # Ejecutor de Hilos
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for video_path in video_files:
            futures.append(
                executor.submit(process_single_video, video_path, args.output_dir, model_path, config)
            )
        
        # Esperar a que terminen
        for f in futures:
            f.result()

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f" PROCESO COMPLETADO EN {total_time:.2f} SEGUNDOS")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
