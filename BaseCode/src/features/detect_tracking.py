import os
import cv2
import gc
import argparse
from ultralytics import YOLO
from sqlalchemy.orm import scoped_session
from sqlalchemy.exc import IntegrityError
import torch
import numpy as np

from src.database.connection import SessionLocal
from src.database.models import FrameObjectDetection, VideoMetadata

# Importar Deep SORT
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    SORT_AVAILABLE = True
except ImportError:
    SORT_AVAILABLE = False
    print("ERROR: Deep SORT no disponible. Instala con: pip install deep-sort-realtime")
    exit(1)

MODEL_PATH = None

def get_model_path():
    """Encuentra la ruta del modelo YOLO"""
    possible_paths = [
        "/app/yolo11x.pt",
        "yolo11x.pt",
        os.path.join(os.getcwd(), "yolo11x.pt"),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    return "yolo11x.pt"

def get_or_create_video(session, video_name):
    """Obtiene o crea un registro de video en la base de datos"""
    video = session.query(VideoMetadata).filter_by(title=video_name).first()
    if not video:
        video = VideoMetadata(title=video_name, duration=0, size=0)
        session.add(video)
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            video = session.query(VideoMetadata).filter_by(title=video_name).first()
    return video

def process_video(video_dir, input_base_dir, conf_threshold=0.4, iou_threshold=0.45):
    """
    Procesa video usando Deep SORT tracker

    Args:
        video_dir: Nombre del directorio del video
        input_base_dir: Directorio base con los frames
        conf_threshold: Umbral de confianza para detecciones (default: 0.25)
        iou_threshold: Umbral IoU para NMS (default: 0.45)
    """
    local_model = None

    try:
        # Inicializar modelo YOLO
        model_path = get_model_path()
        print(f"   Cargando modelo: {model_path}")
        local_model = YOLO(model_path)
        local_model.verbose = False

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        local_model.to(device)
        print(f"    Usando dispositivo: {device}")

        frame_dir = os.path.join(input_base_dir, video_dir)
        if not os.path.isdir(frame_dir):
            return None

        frames = sorted([f for f in os.listdir(frame_dir) if f.endswith(".jpg")])
        if not frames:
            return None

        db = scoped_session(SessionLocal)
        video = get_or_create_video(db, video_dir)

        # ========== CONFIGURACIÓN DEEP SORT ==========
        # Parámetros de Deep SORT (más avanzado que SORT clásico)
        deep_sort_config = {
            "max_age": 30,           # Frames que un track sobrevive sin detección
            "n_init": 3,             # Número de detecciones consecutivas para iniciar track
            "nms_max_overlap": 1.0,  # Máximo overlap para NMS
            "max_cosine_distance": 0.2,  # Distancia máxima para matching
            "nn_budget": 100         # Tamaño del budget para appearance descriptors
        }

        # Inicializar Deep SORT tracker
        tracker = DeepSort(
            max_age=deep_sort_config["max_age"],
            n_init=deep_sort_config["n_init"],
            nms_max_overlap=deep_sort_config["nms_max_overlap"],
            max_cosine_distance=deep_sort_config["max_cosine_distance"],
            nn_budget=deep_sort_config["nn_budget"],
            override_track_class=None,  # Usar clase de track por defecto
            embedder="mobilenet",  # o "clip" para mejor precisión (más lento)
            half=True,  # Usar half precision si está disponible
            bgr=True,   # Las imágenes están en formato BGR (OpenCV)
            embedder_gpu=True if device == 'cuda' else False
        )

        print(f"   Deep SORT configurado: max_age={deep_sort_config['max_age']}, n_init={deep_sort_config['n_init']}")
        print(f"   Cosine distance: {deep_sort_config['max_cosine_distance']}")
        # ===========================================================================

        BATCH_SIZE = 50
        detections_batch = []

        track_history = {}
        frame_count = 0
        total_detections = 0

        print(f"   Procesando {len(frames)} frames...")

        for idx, frame_name in enumerate(frames):
            frame_path = os.path.join(frame_dir, frame_name)
            if not os.path.exists(frame_path):
                continue

            frame = cv2.imread(frame_path)
            if frame is None:
                continue

            try:
                frame_num = int(frame_name.split("_")[1].split(".")[0])
            except (IndexError, ValueError):
                continue

            frame_count += 1

            # ========== DETECCIÓN CON YOLO ==========
            results = local_model.predict(
                frame,
                conf=conf_threshold,
                iou=iou_threshold,
                classes=[0],  # Solo personas (clase 0)
                verbose=False,
                device=device
            )

            # ========== PREPARAR DETECCIONES PARA DEEP SORT ==========
            # Formato requerido por Deep SORT: lista de detecciones [([x1, y1, x2, y2], conf, class_name), ...]
            detections = []

            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()

                # Construir lista de detecciones para Deep SORT
                for box, conf in zip(boxes, confidences):
                    x1, y1, x2, y2 = box[:4]
                    # Convertir a enteros y asegurar coordenadas válidas
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    # Asegurar que las coordenadas estén dentro de la imagen
                    height, width = frame.shape[:2]
                    x1 = max(0, min(x1, width))
                    y1 = max(0, min(y1, height))
                    x2 = max(0, min(x2, width))
                    y2 = max(0, min(y2, height))

                    if x2 > x1 and y2 > y1:  # Solo agregar detecciones válidas
                        detections.append(([x1, y1, x2, y2], conf, "person"))

                total_detections += len(boxes)

            # ========== ACTUALIZAR DEEP SORT TRACKER ==========
            # Deep SORT devuelve: lista de tracks con atributos como:
            # track.track_id, track.to_tlbr() (bounding box), track.confidence, etc.
            tracks = tracker.update_tracks(detections, frame=frame)

            # ========== PROCESAR TRACKS ==========
            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                bbox = track.to_tlbr()  # [x1, y1, x2, y2]

                # Verificar que el bbox sea válido
                if len(bbox) < 4:
                    continue

                x1, y1, x2, y2 = bbox[:4]

                # Registrar en historial
                if track_id not in track_history:
                    track_history[track_id] = {
                        'first_frame': frame_num,
                        'last_frame': frame_num,
                        'detections': 0
                    }

                track_history[track_id]['last_frame'] = frame_num
                track_history[track_id]['detections'] += 1

                # Guardar en base de datos
                detections_batch.append(FrameObjectDetection(
                    video_id=video.video_id,
                    frame_number=frame_num,
                    track_id=track_id,
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2)
                ))

            # Limpiar memoria
            del frame
            gc.collect()

            # Commit por lotes
            if len(detections_batch) >= BATCH_SIZE:
                try:
                    db.bulk_save_objects(detections_batch)
                    db.commit()
                    detections_batch = []
                except Exception as e:
                    db.rollback()
                    print(f"    Error en batch commit: {str(e)}")

            # Progress cada 100 frames
            if (idx + 1) % 100 == 0:
                print(f"   Progreso: {idx + 1}/{len(frames)} frames procesados...")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Commit final
        if detections_batch:
            try:
                db.bulk_save_objects(detections_batch)
                db.commit()
            except Exception as e:
                db.rollback()
                print(f"    Error en commit final: {str(e)}")

        # ========== ESTADÍSTICAS FINALES ==========
        total_tracks_saved = sum(t['detections'] for t in track_history.values())
        print(f"\n   Estadísticas de {video_dir}:")
        print(f"      Frames procesados: {frame_count}")
        print(f"      Detecciones totales: {total_detections}")
        print(f"      Personas únicas (IDs): {len(track_history)}")
        print(f"      Tracks guardados en DB: {total_tracks_saved}")

        if track_history:
            avg_detections = total_tracks_saved / len(track_history)
            print(f"      Promedio detecciones/persona: {avg_detections:.1f}")

            # Mostrar info de los tracks más largos
            sorted_tracks = sorted(
                track_history.items(),
                key=lambda x: x[1]['detections'],
                reverse=True
            )[:5]

            print(f"      Top 5 tracks más largos:")
            for tid, info in sorted_tracks:
                duration = info['last_frame'] - info['first_frame']
                print(f"        • ID {tid}: {info['detections']} detecciones, duración {duration} frames")

        return True

    except Exception as e:
        print(f"\n   Error procesando video {video_dir}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        if local_model is not None:
            del local_model
        if 'db' in locals():
            db.remove()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def process_batch(video_dirs, input_base_dir, conf_threshold, iou_threshold):
    """Procesa un batch de videos secuencialmente"""
    for video_dir in video_dirs:
        print(f"\n{'='*70}")
        print(f" Procesando: {video_dir}")
        print(f"{'='*70}")

        try:
            result = process_video(video_dir, input_base_dir, conf_threshold, iou_threshold)
            if result:
                print(f" Completado: {video_dir}")
            else:
                print(f"  Sin resultados: {video_dir}")
        except Exception as e:
            print(f" Error: {video_dir} - {str(e)}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

def main(input_base_dir, folders, device='cpu', conf_threshold=0.25, iou_threshold=0.45):
    try:
        print("\n" + "="*70)
        print(" YOLO11 + Deep SORT Object Tracking")
        print(" Basado en: github.com/levan92/deep_sort_realtime")
        print("="*70)

        if not SORT_AVAILABLE:
            print(" Deep SORT no está disponible. Abortando.")
            return

        print(f" Dispositivo: {device}")
        print(f" Confianza: {conf_threshold}")
        print(f" IoU threshold: {iou_threshold}")
        print(f" Videos a procesar: {len(folders)}")
        print(f" Directorio base: {input_base_dir}")
        print("="*70)

        # Determinar batch_size
        if device.lower() == 'cuda':
            try:
                total_mem = torch.cuda.get_device_properties(0).total_memory
                allocated_mem = torch.cuda.memory_allocated(0)
                reserved_mem = torch.cuda.memory_reserved(0)

                free_mem = total_mem - (allocated_mem + reserved_mem)
                available_mem = free_mem * 0.85  # 15% margen de seguridad

                mem_per_item = 440 * 1024 * 1024
                batch_size = max(1, int(available_mem // mem_per_item))
            except:
                batch_size = 2
        else:
            batch_size = 2

        print(f" Batch size: {batch_size}\n")

        # Procesar en batches
        for i in range(0, len(folders), batch_size):
            batch = folders[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(folders) + batch_size - 1) // batch_size

            print(f"\n{'='*70}")
            print(f" BATCH {batch_num}/{total_batches}: {', '.join(batch)}")
            print(f"{'='*70}")

            process_batch(batch, input_base_dir, conf_threshold, iou_threshold)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print("\n" + "="*70)
        print(" PROCESAMIENTO COMPLETADO EXITOSAMENTE")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n Error durante el procesamiento: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available():
            print(f" GPU disponible: {torch.cuda.get_device_name(0)}")
        else:
            print("  Usando CPU")
    except ImportError:
        device = 'cpu'
        print("  PyTorch no disponible, usando CPU")

    parser = argparse.ArgumentParser(
        description="Detección y seguimiento de personas con YOLO11 + Deep SORT"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directorio base con subcarpetas de frames"
    )
    parser.add_argument(
        "--folders",
        nargs='+',
        required=True,
        help="Lista de nombres de carpetas a analizar"
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.25,
        help="Umbral de confianza para detecciones (default: 0.25)"
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.45,
        help="Umbral IoU para NMS (default: 0.45)"
    )

    args = parser.parse_args()

    main(
        args.input_dir,
        args.folders,
        device=device,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold
    )
