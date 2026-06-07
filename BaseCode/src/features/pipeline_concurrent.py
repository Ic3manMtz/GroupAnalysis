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
    def __init__(self, distance_threshold=100, min_frames=15, pause_threshold=2.0):
        """
        Args:
            distance_threshold: Distancia máxima en px para considerar un par cercano
            min_frames: Frames consecutivos para confirmar un grupo
            pause_threshold: Velocidad en px/frame por debajo de la cual se considera pausa
        """
        self.distance_threshold = distance_threshold
        self.min_frames = min_frames
        self.pause_threshold = pause_threshold

        self.next_group_id = 1
        self.pair_frames = defaultdict(int)
        self.group_assignments = {}
        self.group_history = defaultdict(lambda: defaultdict(int))

        # --- Nuevas estructuras para estadísticas ---
        # Historial de centroides por grupo: group_id -> [(frame_idx, cx, cy), ...]
        self.group_centroid_history = defaultdict(list)

        # Frame en que cada grupo apareció por primera vez
        self.group_first_frame = {}

        # Frame del último update donde cada grupo estuvo activo
        self.group_last_frame = {}

        # Acumulado de frames en pausa por grupo
        self.group_pause_frames = defaultdict(int)

        # Tamaño (miembros simultáneos) registrado en cada frame: group_id -> [size, ...]
        self.group_size_history = defaultdict(list)

        # Frame actual (se actualiza en cada llamada a update)
        self._current_frame = 0

    def calculate_distance(self, pos1, pos2):
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def calculate_centroid(self, box):
        return ((box['x1'] + box['x2']) / 2, (box['y1'] + box['y2']) / 2)

    def update(self, detections, frame_idx=None):
        """
        Actualiza el estado de grupos.

        Args:
            detections: lista de dicts con track_id, x1, y1, x2, y2
            frame_idx: número de frame actual (para registrar tiempos exactos)
        """
        if frame_idx is not None:
            self._current_frame = frame_idx

        current_centroids = {det['track_id']: self.calculate_centroid(det) for det in detections}
        track_ids = list(current_centroids.keys())
        current_pairs = set()

        # Detección de pares cercanos
        for i in range(len(track_ids)):
            for j in range(i + 1, len(track_ids)):
                id1, id2 = track_ids[i], track_ids[j]
                dist = self.calculate_distance(current_centroids[id1], current_centroids[id2])
                if dist < self.distance_threshold:
                    current_pairs.add(tuple(sorted([id1, id2])))

        # Persistencia temporal de pares
        for pair in current_pairs:
            self.pair_frames[pair] += 1
        for pair in set(self.pair_frames.keys()) - current_pairs:
            self.pair_frames[pair] = 0

        confirmed_pairs = {pair for pair, frames in self.pair_frames.items() if frames >= self.min_frames}

        # Construcción de grupos por componentes conectados
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

        # Asignación de IDs de grupo
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

        # --- Actualizar historial estadístico ---
        unique_groups = defaultdict(list)
        for tid, gid in self.group_assignments.items():
            unique_groups[gid].append(tid)

        output = []
        for gid, members in unique_groups.items():
            if len(members) < 2:
                continue

            # Calcular centroide del grupo en este frame
            centers = []
            for tid in members:
                if tid in current_centroids:
                    centers.append(current_centroids[tid])

            if not centers:
                continue

            cx = float(np.mean([c[0] for c in centers]))
            cy = float(np.mean([c[1] for c in centers]))

            # Registrar frame de inicio
            if gid not in self.group_first_frame:
                self.group_first_frame[gid] = self._current_frame

            # Actualizar frame final
            self.group_last_frame[gid] = self._current_frame

            # Registrar posición en historial
            self.group_centroid_history[gid].append((self._current_frame, cx, cy))

            # Registrar tamaño real (miembros simultáneos) en este frame
            self.group_size_history[gid].append(len(members))

            # Detectar pausa: comparar con posición anterior
            history = self.group_centroid_history[gid]
            if len(history) >= 2:
                prev_frame, prev_cx, prev_cy = history[-2]
                frame_delta = self._current_frame - prev_frame
                if frame_delta > 0:
                    displacement = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                    velocity = displacement / frame_delta  # px/frame
                    if velocity < self.pause_threshold:
                        self.group_pause_frames[gid] += 1

            output.append({'group_id': gid, 'members': members, 'size': len(members)})

        return output

    def get_group_stats(self, fps=30.0):
        """
        Calcula las estadísticas finales de todos los grupos detectados.

        Args:
            fps: frames por segundo del video (para convertir frames a segundos)

        Returns:
            dict: group_id -> dict con todas las métricas estadísticas
        """
        stats = {}

        all_group_ids = set(self.group_first_frame.keys())

        for gid in all_group_ids:
            history = self.group_centroid_history.get(gid, [])

            if len(history) < 2:
                continue

            first_frame = self.group_first_frame[gid]
            last_frame = self.group_last_frame[gid]
            duration_frames = last_frame - first_frame + 1
            duration_seconds = duration_frames / fps

            # --- Cardinalidad ---
            # Número máximo de miembros simultáneos que tuvo el grupo en un solo frame
            size_history = self.group_size_history.get(gid, [])
            cardinality = int(max(size_history)) if size_history else 0

            # --- Velocidades frame a frame ---
            velocities = []
            for i in range(1, len(history)):
                f0, x0, y0 = history[i - 1]
                f1, x1, y1 = history[i]
                frame_delta = f1 - f0
                if frame_delta > 0:
                    displacement = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
                    vel = displacement / frame_delta  # px/frame
                    velocities.append(vel)

            avg_velocity_px = float(np.mean(velocities)) if velocities else 0.0
            max_velocity_px = float(np.max(velocities)) if velocities else 0.0

            # --- Tiempo de pausa ---
            pause_frames = self.group_pause_frames.get(gid, 0)
            pause_seconds = pause_frames / fps
            pause_ratio = pause_frames / duration_frames if duration_frames > 0 else 0.0

            # --- Longitud de vuelo ---
            # Suma de segmentos entre puntos de recorrido (distancia total recorrida)
            flight_length_px = 0.0
            if len(history) >= 2:
                coords = np.array([(cx, cy) for _, cx, cy in history])
                diffs = np.diff(coords, axis=0)
                flight_length_px = float(np.sum(np.linalg.norm(diffs, axis=1)))

            # Longitud de vuelo neta (desplazamiento directo inicio->fin)
            net_displacement_px = 0.0
            if len(history) >= 2:
                _, x_start, y_start = history[0]
                _, x_end, y_end = history[-1]
                net_displacement_px = float(np.sqrt((x_end - x_start)**2 + (y_end - y_start)**2))

            # Índice de rectitud: qué tan directo fue el recorrido (1 = línea recta)
            straightness = (net_displacement_px / flight_length_px) if flight_length_px > 0 else 0.0

            stats[gid] = {
                # Cardinalidad
                'cardinality': cardinality,
                'members': list(self.group_history[gid].keys()),

                # Tiempo de permanencia
                'first_frame': first_frame,
                'last_frame': last_frame,
                'duration_frames': duration_frames,
                'duration_seconds': duration_seconds,

                # Tiempo de pausa
                'pause_frames': pause_frames,
                'pause_seconds': pause_seconds,
                'pause_ratio': pause_ratio,  # fracción del tiempo total en pausa

                # Longitud de vuelo
                'flight_length_px': flight_length_px,
                'net_displacement_px': net_displacement_px,
                'straightness_index': straightness,

                # Velocidad
                'avg_velocity_px_per_frame': avg_velocity_px,
                'max_velocity_px_per_frame': max_velocity_px,
                'avg_velocity_px_per_sec': avg_velocity_px * fps,
                'velocities_raw': velocities,  # para distribución posterior
            }

        return stats


# ==========================================
# FUNCION WORKER (PROCESA UN SOLO VIDEO)
# ==========================================
def process_single_video(video_path, output_stats_dir, model_path, config):
    """
    Función que se ejecutará en cada hilo.
    """
    video_name = os.path.basename(video_path)
    safe_print(f"--> Iniciando hilo para: {video_name}")

    db = SessionLocal()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        model = YOLO(model_path)
        model.to(device)
    except Exception as e:
        safe_print(f"Error cargando modelo para {video_name}: {e}")
        return

    # Registrar video en BD
    video_record = None
    fps = 30.0  # valor por defecto

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            safe_print(f"Error abriendo video: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps else 0
        size_mb = os.path.getsize(video_path) / (1024 * 1024)

        with db_lock:
            video_record = db.query(VideoMetadata).filter_by(title=video_name).first()
            if not video_record:
                video_record = VideoMetadata(title=video_name, duration=duration, size=size_mb)
                db.add(video_record)
                db.commit()
                safe_print(f"    [BD] Nuevo registro creado: {video_name}")
            else:
                safe_print(f"    [BD] Video ya existente: {video_name}")

    except Exception as e:
        safe_print(f"Error inicializando DB para {video_name}: {e}")
        db.close()
        return

    # Inicializar trackers
    tracker = DeepSort(
        max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.2,
        nn_budget=100, embedder="mobilenet", half=True, bgr=True,
        embedder_gpu=(device == 'cuda')
    )

    group_tracker = GroupTracker(
        distance_threshold=config['group_dist'],
        min_frames=config['min_frames'],
        pause_threshold=config.get('pause_threshold', 2.0)
    )

    # Bucle de procesamiento
    frame_idx = 0
    detections_batch = []
    stats = {"groups": 0}
    BATCH_SIZE = 50

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # --- DETECCIÓN ---
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
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                x1, y1, x2, y2 = track.to_ltrb()

                detections_batch.append(FrameObjectDetection(
                    video_id=video_record.video_id, frame_number=frame_idx, track_id=track_id,
                    x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)
                ))

                current_frame_tracks.append({
                    'track_id': track_id, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                })

            # --- GRUPOS ---
            if current_frame_tracks:
                # Pasamos frame_idx para que el tracker registre tiempos exactos
                groups_found = group_tracker.update(current_frame_tracks, frame_idx=frame_idx)
                stats["groups"] = len(groups_found)

                for grp in groups_found:
                    centers = []
                    for tid in grp['members']:
                        t = next((t for t in current_frame_tracks if t['track_id'] == tid), None)
                        if t:
                            centers.append([(t['x1'] + t['x2']) / 2, (t['y1'] + t['y2']) / 2])

                    cx, cy, disp = 0, 0, 0
                    if centers:
                        centers = np.array(centers)
                        cx, cy = np.mean(centers, axis=0)
                        disp = float(np.mean(np.linalg.norm(centers - np.array([cx, cy]), axis=1)))

                    group_obj = GroupDetection(
                        video_id=video_record.video_id, frame_number=frame_idx,
                        group_id=grp['group_id'], center_x=float(cx), center_y=float(cy),
                        size=grp['size'], dispersion=disp, avg_velocity=0.0, velocity_std=0.0
                    )
                    db.add(group_obj)
                    db.flush()

                    for mid in grp['members']:
                        db.add(GroupMember(group_detection_id=group_obj.id, track_id=mid))

            # --- BATCH SAVE ---
            if len(detections_batch) >= BATCH_SIZE:
                db.bulk_save_objects(detections_batch)
                db.commit()
                detections_batch = []

            if frame_idx % 100 == 0:
                safe_print(f"   [{video_name[:10]}...] Frame {frame_idx} | Grupos activos: {stats['groups']}")

        # Commit final
        if detections_batch:
            db.bulk_save_objects(detections_batch)
        db.commit()

        # Calcular estadísticas de grupos en memoria y generar reporte
        group_stats = group_tracker.get_group_stats(fps=fps)
        generate_report(db, video_record, output_stats_dir, group_stats, fps)

    except Exception as e:
        safe_print(f"ERROR FATAL en {video_name}: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        db.close()
        del model
        del tracker
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
        safe_print(f"--> Finalizado: {video_name}")


# ==========================================
# GENERACIÓN DE REPORTE ESTADÍSTICO
# ==========================================
def _distribution_summary(values, unit=""):
    """
    Genera un resumen estadístico de una lista de valores.
    Retorna un dict con las métricas de distribución.
    """
    if not values:
        return None

    arr = np.array(values, dtype=float)
    return {
        'n': len(arr),
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'p25': float(np.percentile(arr, 25)),
        'p75': float(np.percentile(arr, 75)),
        'unit': unit
    }

def _write_distribution(f, label, dist):
    """Escribe un bloque de distribución estadística en el archivo de reporte."""
    if dist is None:
        f.write(f"  {label}: sin datos\n")
        return

    u = dist['unit']
    f.write(f"  {label} (n={dist['n']}):\n")
    f.write(f"    Media    : {dist['mean']:.3f} {u}\n")
    f.write(f"    Mediana  : {dist['median']:.3f} {u}\n")
    f.write(f"    Desv. Std: {dist['std']:.3f} {u}\n")
    f.write(f"    Mínimo   : {dist['min']:.3f} {u}\n")
    f.write(f"    Máximo   : {dist['max']:.3f} {u}\n")
    f.write(f"    P25      : {dist['p25']:.3f} {u}\n")
    f.write(f"    P75      : {dist['p75']:.3f} {u}\n")


def generate_report(db, video, output_folder, group_stats, fps=30.0):
    """
    Genera reporte estadístico completo por grupo y distribuciones agregadas.

    Args:
        db: sesión de SQLAlchemy
        video: objeto VideoMetadata
        output_folder: carpeta de salida
        group_stats: dict con estadísticas calculadas por GroupTracker.get_group_stats()
        fps: frames por segundo del video
    """
    try:
        os.makedirs(output_folder, exist_ok=True)
        report_path = os.path.join(output_folder, f"REPORTE_{video.title}.txt")

        # Datos globales desde BD
        total_frames_row = db.query(FrameObjectDetection.frame_number)\
            .filter_by(video_id=video.video_id)\
            .order_by(FrameObjectDetection.frame_number.desc()).first()
        last_frame = total_frames_row[0] if total_frames_row else 0

        unique_people = db.query(FrameObjectDetection.track_id)\
            .filter_by(video_id=video.video_id).distinct().count()

        # Recolectar valores para distribuciones agregadas
        all_cardinalities = []
        all_durations_sec = []
        all_pause_seconds = []
        all_pause_ratios = []
        all_flight_lengths = []
        all_net_displacements = []
        all_straightness = []
        all_avg_velocities = []
        all_max_velocities = []

        for gid, s in group_stats.items():
            all_cardinalities.append(s['cardinality'])
            all_durations_sec.append(s['duration_seconds'])
            all_pause_seconds.append(s['pause_seconds'])
            all_pause_ratios.append(s['pause_ratio'])
            all_flight_lengths.append(s['flight_length_px'])
            all_net_displacements.append(s['net_displacement_px'])
            all_straightness.append(s['straightness_index'])
            all_avg_velocities.append(s['avg_velocity_px_per_sec'])
            all_max_velocities.append(s['max_velocity_px_per_frame'])

        with open(report_path, "w", encoding="utf-8") as f:

            # ── ENCABEZADO ──────────────────────────────────────────────
            f.write(f"REPORTE ESTADÍSTICO - {video.title}\n")
            f.write(f"{'='*60}\n")
            f.write(f"FPS del video        : {fps:.2f}\n")
            f.write(f"Duración total       : {last_frame} frames "
                    f"({last_frame / fps:.1f} segundos)\n")
            f.write(f"Personas únicas      : {unique_people}\n")
            f.write(f"Grupos detectados    : {len(group_stats)}\n")
            f.write(f"{'='*60}\n\n")

            # ── DETALLE POR GRUPO ────────────────────────────────────────
            f.write("DETALLE POR GRUPO\n")
            f.write(f"{'-'*60}\n")

            for gid, s in sorted(group_stats.items()):
                f.write(f"\nGrupo {gid}\n")
                f.write(f"  Miembros (track IDs)  : {s['members']}\n")

                # Cardinalidad
                f.write(f"  Cardinalidad          : {s['cardinality']} personas\n")

                # Tiempo de permanencia
                f.write(f"  Permanencia           : "
                        f"frames {s['first_frame']} → {s['last_frame']} "
                        f"({s['duration_frames']} frames / "
                        f"{s['duration_seconds']:.2f} seg)\n")

                # Tiempo de pausa
                f.write(f"  Tiempo en pausa       : "
                        f"{s['pause_frames']} frames / "
                        f"{s['pause_seconds']:.2f} seg "
                        f"({s['pause_ratio']*100:.1f}% del tiempo)\n")

                # Longitud de vuelo
                f.write(f"  Longitud de vuelo     : {s['flight_length_px']:.1f} px\n")
                f.write(f"  Desplazamiento neto   : {s['net_displacement_px']:.1f} px\n")
                f.write(f"  Índice de rectitud    : {s['straightness_index']:.3f} "
                        f"(1=línea recta, 0=sin avance)\n")

                # Velocidad
                f.write(f"  Velocidad promedio    : "
                        f"{s['avg_velocity_px_per_frame']:.2f} px/frame | "
                        f"{s['avg_velocity_px_per_sec']:.2f} px/seg\n")
                f.write(f"  Velocidad máxima      : "
                        f"{s['max_velocity_px_per_frame']:.2f} px/frame\n")

            # ── DISTRIBUCIONES AGREGADAS ─────────────────────────────────
            f.write(f"\n{'='*60}\n")
            f.write("DISTRIBUCIONES ESTADÍSTICAS (todos los grupos)\n")
            f.write(f"{'='*60}\n\n")

            _write_distribution(f, "Cardinalidad (personas por grupo)",
                                _distribution_summary(all_cardinalities, "personas"))

            f.write("\n")
            _write_distribution(f, "Tiempo de permanencia",
                                _distribution_summary(all_durations_sec, "segundos"))

            f.write("\n")
            _write_distribution(f, "Tiempo de pausa",
                                _distribution_summary(all_pause_seconds, "segundos"))

            f.write("\n")
            _write_distribution(f, "Ratio de pausa (fracción del tiempo en pausa)",
                                _distribution_summary(all_pause_ratios, ""))

            f.write("\n")
            _write_distribution(f, "Longitud de vuelo (recorrido total)",
                                _distribution_summary(all_flight_lengths, "px"))

            f.write("\n")
            _write_distribution(f, "Desplazamiento neto (inicio → fin)",
                                _distribution_summary(all_net_displacements, "px"))

            f.write("\n")
            _write_distribution(f, "Índice de rectitud del recorrido",
                                _distribution_summary(all_straightness, ""))

            f.write("\n")
            _write_distribution(f, "Velocidad promedio",
                                _distribution_summary(all_avg_velocities, "px/seg"))

            f.write("\n")
            _write_distribution(f, "Velocidad máxima",
                                _distribution_summary(all_max_velocities, "px/frame"))

            f.write(f"\n{'='*60}\n")
            f.write("FIN DEL REPORTE\n")

        safe_print(f"    [REPORT] Generado en: {report_path}")

    except Exception as e:
        safe_print(f"Error generando reporte para {video.title}: {e}")
        import traceback
        traceback.print_exc()


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
    parser.add_argument("--pause_threshold", type=float, default=2.0,
                        help="Velocidad en px/frame por debajo de la cual se considera pausa (default: 2.0)")

    args = parser.parse_args()

    video_files = glob.glob(os.path.join(args.input_dir, "*.[mM][pP]4"))
    if not video_files:
        print(f"No se encontraron videos MP4 en {args.input_dir}")
        return

    print(f"\n{'='*60}")
    print(f" INICIANDO PROCESAMIENTO CONCURRENTE")
    print(f" Videos encontrados: {len(video_files)}")
    print(f" Hilos simultáneos : {args.max_workers}")
    print(f" GPU Disponible    : {torch.cuda.is_available()}")
    print(f" Pause threshold   : {args.pause_threshold} px/frame")
    print(f"{'='*60}\n")

    model_path = "yolo11x.pt"

    config = {
        'conf': args.conf,
        'group_dist': args.group_dist,
        'min_frames': args.min_frames,
        'pause_threshold': args.pause_threshold,
    }

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for video_path in video_files:
            futures.append(
                executor.submit(process_single_video, video_path, args.output_dir, model_path, config)
            )
        for f in futures:
            f.result()

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f" PROCESO COMPLETADO EN {total_time:.2f} SEGUNDOS")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()