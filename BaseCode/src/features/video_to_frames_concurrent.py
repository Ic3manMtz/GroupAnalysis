import cv2
import os
import glob
from tqdm import tqdm
import threading
import argparse
from sqlalchemy import or_
from src.database.connection import SessionLocal
from src.database.db_crud import VideoCRUD

def get_video_size(video_path):
    """Obtiene el tamaño del archivo de video en megabytes"""
    return os.path.getsize(video_path) / (1024 * 1024)  # Convertir a MB


def process_video(video_path, output_folder, position, lock, disable_progress=False):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_folder = os.path.join(output_folder, video_name)
    os.makedirs(video_output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error al abrir video: {video_path}, saltando...")
        return

    # Obtener metadatos del video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    video_size = get_video_size(video_path)

    frame_count = 0
    # Calcular sampling_rate basado en FPS para obtener 2 frames por segundo
    # Si fps = 60, sampling_rate = 30 (extraer 1 de cada 30 frames = 2 frames/seg)
    # Si fps = 30, sampling_rate = 15 (extraer 1 de cada 15 frames = 2 frames/seg)
    frames_per_second = 25
    sampling_rate = int(fps / frames_per_second) if fps > 0 else 30
    saved_frames = 0

    db = SessionLocal()
    video_crud = VideoCRUD(db)
    try:
        # Usar bloqueo para evitar condiciones de carrera
        with lock:
            # Verificar si el video ya existe
            video_record = video_crud.get_video_by_title(video_name)
            if not video_record:
                # Si no existe, crear el registro
                video_record = video_crud.create_video(
                    title=video_name,
                    duration=duration,
                    size=video_size
                )

        # Configurar la barra de progreso con position fija y dynamic_ncols
        pbar = tqdm(
            total=total_frames,
            desc=f"(FPS:{frames_per_second}) Extrayendo {video_name[:15]}...",
            unit="frame",
            position=position,
            leave=False,  # Cambiado a False para limpiar la barra al terminar
            disable=disable_progress,
            dynamic_ncols=True  # Ajustar al ancho de la terminal
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sampling_rate == 0:
                frame_file = os.path.join(
                    video_output_folder,
                    f"frame_{frame_count:06d}.jpg"
                )
                cv2.imwrite(frame_file, frame)
                saved_frames += 1

            frame_count += 1
            pbar.update(1)

        pbar.close()  # Cerrar la barra adecuadamente
        print(f"{video_name[:15]} ({frame_count} frames, {saved_frames} guardados)")

    except Exception as e:
        print(f"Error procesando video {video_name}: {str(e)}")
    finally:
        db.close()
        cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from videos.")
    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="Path to the directory containing video files."
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Path to the directory where frames will be saved."
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars for cleaner output."
    )
    args = parser.parse_args()

    video_dir = args.video_dir
    output_folder = args.output_folder

    if not os.path.exists(video_dir):
        raise FileNotFoundError(f"⚠️ Carpeta de videos no encontrada: {video_dir}")

    # Crear directorio principal para frames
    frames_dir = os.path.join(output_folder, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    video_files = glob.glob(os.path.join(video_dir, "*.[mM][pP]4"))
    if not video_files:
        print("No se encontraron videos MP4 en la carpeta")
        exit()

    # Crear un Lock para sincronización entre hilos
    lock = threading.Lock()

    # Calcula número óptimo de hilos
    optimal_threads = 5  # Ajusta este valor según tu sistema

    # Procesar videos en hilos (con límite)
    threads = []
    for idx, video in enumerate(video_files):
        if len(threads) >= optimal_threads:
            # Esperar a que haya disponibilidad
            for t in threads:
                t.join()
            threads = []

        t = threading.Thread(
            target=process_video,
            args=(video, frames_dir, idx % optimal_threads, lock, args.no_progress)
        )
        threads.append(t)
        t.start()

    # Esperar los hilos restantes
    for t in threads:
        t.join()

    # os.system('cls' if os.name == 'nt' else 'clear')
    print("\nProceso completado para todos los videos!")
