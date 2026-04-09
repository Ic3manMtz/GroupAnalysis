import os
import cv2
import argparse
from tqdm import tqdm
from sqlalchemy.orm import sessionmaker
from src.database.connection import SessionLocal
from src.database.models import FrameObjectDetection, VideoMetadata


def draw_rounded_rect(img, pt1, pt2, color, thickness, corner_radius):
    """Dibuja un rectángulo con bordes redondeados"""
    x1, y1 = pt1
    x2, y2 = pt2

    # Dibujar rectángulo principal
    cv2.rectangle(img, (x1 + corner_radius, y1),
                  (x2 - corner_radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + corner_radius),
                  (x2, y2 - corner_radius), color, thickness)

    # Dibujar esquinas redondeadas
    cv2.ellipse(img, (x1 + corner_radius, y1 + corner_radius),
                (corner_radius, corner_radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - corner_radius, y1 + corner_radius),
                (corner_radius, corner_radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + corner_radius, y2 - corner_radius),
                (corner_radius, corner_radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - corner_radius, y2 - corner_radius),
                (corner_radius, corner_radius), 0, 0, 90, color, thickness)


def get_color_from_track_id(track_id):
    """Genera un color consistente basado en el track_id"""
    # Usar una paleta de colores vibrantes
    colors = [
        (0, 255, 255),  # Amarillo
        (255, 0, 255),  # Magenta
        (0, 165, 255),  # Naranja
        (255, 0, 0),  # Azul
        (0, 255, 0),  # Verde
        (0, 0, 255),  # Rojo
        (255, 255, 0),  # Cian
        (128, 0, 128)  # Púrpura
    ]
    return colors[track_id % len(colors)]


def visualize_detections(video_name, input_frames_dir, output_video_path, fps=30):
    """Genera un video con las detecciones visualizadas"""
    db = SessionLocal()

    try:
        # Obtener el video de la base de datos
        video = db.query(VideoMetadata).filter_by(title=video_name).first()
        if not video:
            print(f"\nVideo {video_name} no encontrado en la base de datos")
            return False

        # Obtener todos los frames para este video
        frames_dir = os.path.join(input_frames_dir, video_name)
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])

        if not frame_files:
            print(f"\nNo se encontraron frames para el video {video_name}")
            return False

        # Obtener las dimensiones del primer frame
        first_frame_path = os.path.join(frames_dir, frame_files[0])
        first_frame = cv2.imread(first_frame_path)
        if first_frame is None:
            print(f"\nNo se pudo leer el primer frame: {first_frame_path}")
            return False

        height, width = first_frame.shape[:2]

        # Configurar el writer de video (mejor calidad)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Configurar barra de progreso para frames
        progress_bar = tqdm(frame_files, desc=f"Procesando {video_name}", unit="frame")

        # Procesar cada frame
        for frame_file in progress_bar:
            try:
                frame_num = int(frame_file.split('_')[1].split('.')[0])
            except (IndexError, ValueError):
                continue

            frame_path = os.path.join(frames_dir, frame_file)
            frame = cv2.imread(frame_path)

            if frame is None:
                continue

            # Obtener todas las detecciones para este frame
            detections = db.query(FrameObjectDetection).filter_by(
                video_id=video.video_id,
                frame_number=frame_num
            ).all()

            # Dibujar cada detección en el frame
            for det in detections:
                # Convertir coordenadas a enteros
                x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)

                # Obtener color basado en track_id
                color = get_color_from_track_id(det.track_id)

                # Grosor de la línea adaptativo al tamaño del frame
                thickness = max(2, int(min(height, width) / 300))
                corner_radius = max(5, int(min(height, width) / 150))

                # Dibujar bounding box con bordes redondeados
                draw_rounded_rect(frame, (x1, y1), (x2, y2), color, thickness, corner_radius)

                # Preparar texto de la etiqueta
                label = f"ID: {det.track_id}"
                font_scale = max(0.5, min(height, width) / 1500)
                font_thickness = max(1, int(min(height, width) / 600))

                # Calcular tamaño del texto para el fondo
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

                # Dibujar fondo semitransparente para el texto
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, y1 - text_height - 10),
                              (x1 + text_width + 10, y1), (0, 0, 0), -1)
                alpha = 0.6  # Factor de transparencia
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                # Dibujar texto
                cv2.putText(frame, label, (x1 + 5, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (255, 255, 255), font_thickness, cv2.LINE_AA)

            # Escribir frame en el video de salida
            out.write(frame)

        out.release()
        progress_bar.close()
        print(f"\nVideo generado: {output_video_path}")
        return True

    except Exception as e:
        print(f"\nError procesando {video_name}: {str(e)}")
        return False
    finally:
        db.close()


def process_videos(input_dir, folders, output_base_dir):
    """Procesa múltiples videos según los parámetros"""
    # Crear directorio de salida si no existe
    output_dir = os.path.join(output_base_dir, "converted_videos")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nIniciando procesamiento de {len(folders)} videos...")

    # Configurar barra de progreso para videos
    video_progress = tqdm(folders, desc="Videos completados", unit="video")

    # Procesar cada video
    for video_name in video_progress:
        video_progress.set_postfix({"video": video_name})

        input_frames_dir = input_dir
        output_video_path = os.path.join(output_dir, f"{video_name}_detections.mp4")

        # Verificar si el video ya fue procesado
        if os.path.exists(output_video_path):
            video_progress.write(f"El video {video_name} ya fue procesado, omitiendo...")
            continue

        visualize_detections(video_name, input_frames_dir, output_video_path)

    video_progress.close()
    print("\nProceso de conversión completado")


def main():
    parser = argparse.ArgumentParser(description="Visualizar detecciones en videos a partir de frames")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directorio base con subcarpetas de frames")
    parser.add_argument("--folders", nargs='+', required=True,
                        help="Lista de nombres de carpetas a analizar")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Directorio base donde se creará la carpeta 'converted_videos'")

    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("Visualización de detecciones en videos")
    print("=" * 50)
    print(f"\nDirectorio de entrada: {args.input_dir}")
    print(f"Carpetas a procesar: {args.folders}")
    print(f"Directorio de salida: {args.output_folder}")
    print("\n" + "=" * 50 + "\n")

    process_videos(args.input_dir, args.folders, args.output_folder)


if __name__ == "__main__":
    main()
