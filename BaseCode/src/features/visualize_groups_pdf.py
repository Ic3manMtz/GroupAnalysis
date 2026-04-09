import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sqlalchemy.orm import sessionmaker
from src.database.connection import SessionLocal
from src.database.models import VideoMetadata, FrameObjectDetection, GroupDetection, GroupMember
import time

def debug_log(message):
    """Función para logging de diagnóstico"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] 🔍 {message}")

def get_video_data(video_name):
    """Obtiene datos del video de forma segura con diagnóstico"""
    debug_log(f"Iniciando carga de datos para: {video_name}")

    db = SessionLocal()
    try:
        debug_log("Buscando video en la base de datos...")
        video = db.query(VideoMetadata).filter_by(title=video_name).first()
        if not video:
            debug_log(f"❌ Video {video_name} no encontrado en la base de datos")
            return None, None, None

        debug_log(f"✅ Video encontrado - ID: {video.video_id}")

        # Obtener grupos
        debug_log("Consultando grupos...")
        groups = db.query(GroupDetection).filter_by(video_id=video.video_id).all()
        debug_log(f"Encontrados {len(groups)} registros de grupos")

        groups_by_frame = defaultdict(list)

        debug_log("Procesando grupos...")
        for i, group in enumerate(groups):
            if i % 100 == 0:  # Log cada 100 grupos
                debug_log(f"Procesando grupo {i}/{len(groups)}")

            members = db.query(GroupMember).filter_by(group_detection_id=group.id).all()
            groups_by_frame[group.frame_number].append({
                'group_id': group.group_id,
                'size': group.size,
                'type': 'couple' if group.size == 2 else 'group',
                'center_x': group.center_x,
                'center_y': group.center_y,
                'members': [m.track_id for m in members]
            })

        debug_log("✅ Grupos procesados correctamente")

        # Obtener detecciones
        debug_log("Consultando detecciones...")
        detections = db.query(FrameObjectDetection).filter_by(video_id=video.video_id).all()
        debug_log(f"Encontradas {len(detections)} detecciones")

        detections_by_frame = defaultdict(dict)

        debug_log("Procesando detecciones...")
        for i, det in enumerate(detections):
            if i % 1000 == 0:  # Log cada 1000 detecciones
                debug_log(f"Procesando detección {i}/{len(detections)}")

            detections_by_frame[det.frame_number][det.track_id] = {
                'x1': det.x1, 'y1': det.y1, 'x2': det.x2, 'y2': det.y2
            }

        debug_log("✅ Detecciones procesadas correctamente")
        debug_log(f"Resumen: {len(groups_by_frame)} frames con grupos, {len(detections_by_frame)} frames con detecciones")

        return groups_by_frame, detections_by_frame, video

    except Exception as e:
        debug_log(f"❌ Error obteniendo datos de la base de datos: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None
    finally:
        db.close()
        debug_log("Conexión a BD cerrada")

def check_frames_directory(frames_dir):
    """Verifica que el directorio de frames exista y tenga archivos"""
    debug_log(f"Verificando directorio: {frames_dir}")

    if not os.path.exists(frames_dir):
        debug_log(f"❌ Directorio no existe: {frames_dir}")
        return False, []

    frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')]
    debug_log(f"Encontrados {len(frame_files)} archivos JPG")

    if len(frame_files) == 0:
        debug_log("❌ No hay archivos JPG en el directorio")
        return False, []

    # Verificar que al menos el primer archivo sea legible
    if frame_files:
        first_frame = os.path.join(frames_dir, frame_files[0])
        debug_log(f"Verificando primer frame: {first_frame}")
        test_frame = cv2.imread(first_frame)
        if test_frame is None:
            debug_log("❌ No se puede leer el primer frame")
            return False, []
        else:
            debug_log(f"✅ Primer frame OK - Dimensiones: {test_frame.shape}")

    return True, sorted(frame_files)

def draw_progress_bar(frame, progress, width=300, height=20, x=10, y=10):
    """Dibuja una barra de progreso en el frame"""
    # Fondo de la barra
    cv2.rectangle(frame, (x, y), (x + width, y + height), (50, 50, 50), -1)

    # Barra de progreso
    progress_width = int(width * progress)
    if progress_width > 0:
        color = (0, int(255 * progress), int(255 * (1 - progress)))
        cv2.rectangle(frame, (x, y), (x + progress_width, y + height), color, -1)

    # Borde
    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 2)

    # Texto de porcentaje
    percent_text = f"{progress*100:.1f}%"
    text_size = cv2.getTextSize(percent_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    text_x = x + (width - text_size[0]) // 2
    text_y = y + height + text_size[1] + 5
    cv2.putText(frame, percent_text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def visualize_groups_pdf(video_name, input_frames_dir, output_video_path, fps=30):
    """Visualiza grupos detectados con el algoritmo del PDF - CON DIAGNÓSTICO"""

    debug_log(f"Iniciando visualización para: {video_name}")

    # PASO 1: Obtener datos de la base de datos
    debug_log("=== PASO 1: Cargando datos de la base de datos ===")
    groups_by_frame, detections_by_frame, video = get_video_data(video_name)

    if not groups_by_frame:
        debug_log("❌ No se pudieron cargar los datos de grupos")
        return False

    # PASO 2: Verificar directorio de frames
    debug_log("=== PASO 2: Verificando directorio de frames ===")
    frames_dir = os.path.join(input_frames_dir, video_name)
    frames_ok, frame_files = check_frames_directory(frames_dir)

    if not frames_ok:
        debug_log("❌ Problemas con el directorio de frames")
        return False

    debug_log(f"✅ Listo para procesar {len(frame_files)} frames")

    # PASO 3: Configurar video writer
    debug_log("=== PASO 3: Configurando video writer ===")
    try:
        first_frame_path = os.path.join(frames_dir, frame_files[0])
        debug_log(f"Leyendo primer frame: {first_frame_path}")
        first_frame = cv2.imread(first_frame_path)

        if first_frame is None:
            debug_log("❌ No se pudo leer el primer frame")
            return False

        height, width = first_frame.shape[:2]
        debug_log(f"Dimensiones del video: {width}x{height}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        debug_log(f"✅ Video writer configurado: {output_video_path}")

    except Exception as e:
        debug_log(f"❌ Error configurando video writer: {str(e)}")
        return False

    # Colores para diferentes grupos
    colors = [
        (0, 255, 255), (255, 0, 255), (0, 165, 255),
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (128, 0, 128), (255, 165, 0),
        (0, 128, 128), (128, 128, 0), (128, 0, 0),
    ]

    # Estadísticas
    total_groups = 0
    total_detections = 0

    debug_log("=== PASO 4: Procesando frames ===")

    try:
        # Barra de progreso principal para frames
        progress_bar = tqdm(frame_files, desc="Procesando frames", unit="frame",
                           bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}')

        start_time = time.time()

        for frame_index, frame_file in enumerate(progress_bar):
            try:
                # Extraer número de frame
                frame_num = int(frame_file.split('_')[1].split('.')[0])
            except (IndexError, ValueError):
                debug_log(f"⚠️  No se pudo extraer número de frame de: {frame_file}")
                continue

            frame_path = os.path.join(frames_dir, frame_file)
            frame = cv2.imread(frame_path)
            if frame is None:
                debug_log(f"⚠️  No se pudo leer frame: {frame_file}")
                continue

            # Calcular progreso actual
            progress = (frame_index + 1) / len(frame_files)

            # Dibujar barra de progreso en el frame
            draw_progress_bar(frame, progress, width=400, height=25, x=10, y=height - 60)

            # Dibujar grupos del frame actual
            if frame_num in groups_by_frame:
                frame_groups = groups_by_frame[frame_num]
                total_groups += len(frame_groups)

                for group in frame_groups:
                    color = colors[group['group_id'] % len(colors)]

                    # Dibujar bounding boxes de miembros
                    member_count = 0
                    for track_id in group['members']:
                        if track_id in detections_by_frame.get(frame_num, {}):
                            det = detections_by_frame[frame_num][track_id]
                            x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])

                            # Bounding box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                            # Etiqueta del track
                            label = f"ID:{track_id}"
                            cv2.putText(frame, label, (x1, max(y1-10, 15)),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            member_count += 1
                            total_detections += 1

                    # Información del grupo
                    if member_count > 0:
                        group_type = "Pareja" if group['size'] == 2 else f"Grupo({group['size']})"
                        info_text = f"G{group['group_id']} {group_type}"
                        text_y = 60 + (group['group_id'] * 25) % (height - 100)
                        cv2.putText(frame, info_text, (10, text_y),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Información del frame
            cv2.putText(frame, f"Frame: {frame_num}", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            group_count = len(groups_by_frame.get(frame_num, []))
            cv2.putText(frame, f"Grupos: {group_count}", (width - 150, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            progress_text = f"Progreso: {progress*100:.1f}%"
            cv2.putText(frame, progress_text, (width - 200, height - 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            out.write(frame)

            # Actualizar descripción de la barra de progreso
            progress_bar.set_description(f"Procesando frames (Grupos: {total_groups})")

        # Calcular tiempo total
        end_time = time.time()
        total_time = end_time - start_time

        out.release()
        progress_bar.close()

        # Mostrar estadísticas finales
        debug_log("=== PROCESO COMPLETADO ===")
        print(f"\n📊 ESTADÍSTICAS FINALES:")
        print(f"   • Frames procesados: {len(frame_files)}")
        print(f"   • Grupos detectados: {total_groups}")
        print(f"   • Detecciones dibujadas: {total_detections}")
        print(f"   • Tiempo total: {total_time:.2f} segundos")
        print(f"   • FPS promedio: {len(frame_files)/total_time:.2f}")
        print(f"   • Archivo de salida: {output_video_path}")
        print(f"✅ ¡Video generado exitosamente!")

        return True

    except Exception as e:
        debug_log(f"❌ Error durante el procesamiento: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Asegurar que el writer se cierre
        if 'out' in locals():
            out.release()
        if 'progress_bar' in locals():
            progress_bar.close()

def main():
    parser = argparse.ArgumentParser(description="Visualización de grupos detectados - CON DIAGNÓSTICO")
    parser.add_argument("--video_name", type=str, required=True,
                       help="Nombre del video a visualizar")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directorio con los frames")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directorio de salida para el video")
    parser.add_argument("--fps", type=int, default=30,
                       help="FPS del video de salida")

    args = parser.parse_args()

    # Crear directorio de salida
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.video_name}_groups_pdf.mp4")

    print(f"\n{'='*60}")
    print(f"🎬 VISUALIZACIÓN CON DIAGNÓSTICO - {args.video_name}")
    print(f"{'='*60}")
    print(f"📂 Directorio de entrada: {args.input_dir}")
    print(f"📂 Directorio de salida: {args.output_dir}")
    print(f"🎯 Archivo de salida: {output_path}")
    print(f"📺 FPS: {args.fps}")
    print(f"{'='*60}\n")

    success = visualize_groups_pdf(
        args.video_name,
        args.input_dir,
        output_path,
        args.fps
    )

    if success:
        print(f"\n🎉 ¡Proceso completado exitosamente!")
    else:
        print(f"\n❌ El proceso falló - Revisar logs anteriores")

    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
