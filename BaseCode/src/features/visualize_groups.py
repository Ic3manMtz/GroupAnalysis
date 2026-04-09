import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from src.database.connection import SessionLocal
from src.database.models import (VideoMetadata, FrameObjectDetection,
                                  GroupDetection, GroupMember)


def generate_color_palette(n_colors):
    """Genera una paleta de colores distinguibles"""
    colors = []
    for i in range(n_colors):
        hue = int(180 * i / n_colors)
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(map(int, color)))
    return colors


def draw_group_visualization(frame, group_info, detections_map, color):
    """Dibuja la visualización de un grupo en el frame"""
    group_members = group_info['members']

    # Dibujar bounding boxes de miembros
    for member_track_id in group_members:
        if member_track_id in detections_map:
            det = detections_map[member_track_id]
            x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])

            # Dibujar bounding box del miembro
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Etiqueta del track
            label = f"ID:{member_track_id}"
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Dibujar elipse alrededor del grupo
    if len(group_members) >= 2:
        centers = []
        for member_track_id in group_members:
            if member_track_id in detections_map:
                det = detections_map[member_track_id]
                center_x = (det['x1'] + det['x2']) / 2
                center_y = (det['y1'] + det['y2']) / 2
                centers.append([center_x, center_y])

        if len(centers) >= 2:
            centers = np.array(centers)

            # Calcular centro del grupo
            group_center = np.mean(centers, axis=0)

            try:
                # Calcular la matriz de covarianza para la elipse
                cov = np.cov(centers.T)

                # Agregar pequeño valor para evitar singularidad
                cov += np.eye(2) * 1e-5

                eigenvalues, eigenvectors = np.linalg.eig(cov)

                # Validar eigenvalues
                if np.all(eigenvalues > 0) and np.all(np.isfinite(eigenvalues)):
                    # Calcular ángulo y ejes
                    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
                    width = 4 * np.sqrt(abs(eigenvalues[0]))
                    height = 4 * np.sqrt(abs(eigenvalues[1]))

                    # Validar que width y height sean finitos y razonables
                    if np.isfinite(width) and np.isfinite(height) and width > 0 and height > 0:
                        # Limitar tamaño máximo de la elipse
                        max_axis = min(frame.shape[0], frame.shape[1]) / 2
                        width = min(width, max_axis)
                        height = min(height, max_axis)

                        # Dibujar elipse
                        cv2.ellipse(frame,
                                   (int(group_center[0]), int(group_center[1])),
                                   (int(width), int(height)),
                                   angle, 0, 360, color, 2)
                else:
                    # Si la elipse no es válida, dibujar círculo simple
                    # Calcular radio como distancia promedio desde el centro
                    distances = np.linalg.norm(centers - group_center, axis=1)
                    radius = int(np.mean(distances) * 2)
                    if radius > 0 and np.isfinite(radius):
                        cv2.circle(frame,
                                 (int(group_center[0]), int(group_center[1])),
                                 min(radius, 200), color, 2)
            except Exception as e:
                # Si falla el cálculo de la elipse, dibujar círculo simple
                distances = np.linalg.norm(centers - group_center, axis=1)
                radius = int(np.mean(distances) * 2) if len(distances) > 0 else 50
                if radius > 0 and np.isfinite(radius):
                    cv2.circle(frame,
                             (int(group_center[0]), int(group_center[1])),
                             min(radius, 200), color, 2)

            # Dibujar centro del grupo
            if np.isfinite(group_center[0]) and np.isfinite(group_center[1]):
                cv2.circle(frame, (int(group_center[0]), int(group_center[1])),
                          5, color, -1)

    # Información del grupo
    features = group_info['features']
    info_text = [
        f"Grupo {group_info['group_id']}",
        f"Personas: {features['size']}",
        f"Velocidad: {features['avg_velocity']:.1f}",
        f"Dispersion: {features['dispersion']:.1f}"
    ]

    # Posición para el texto (esquina superior derecha del primer miembro)
    text_x = int(group_info['features']['center_x'])
    text_y = int(group_info['features']['center_y']) - 80

    for i, text in enumerate(info_text):
        y_pos = text_y + i * 20

        # Fondo semi-transparente
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        overlay = frame.copy()
        cv2.rectangle(overlay,
                     (text_x - 5, y_pos - text_height - 2),
                     (text_x + text_width + 5, y_pos + 2),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Texto
        cv2.putText(frame, text, (text_x, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return frame


def visualize_groups_video(video_name, input_frames_dir, output_video_path, fps=30):
    """Genera un video con las detecciones de grupos visualizadas"""
    db = SessionLocal()

    try:
        # Obtener video
        video = db.query(VideoMetadata).filter_by(title=video_name).first()
        if not video:
            print(f"Video {video_name} no encontrado")
            return False

        # Obtener frames
        frames_dir = os.path.join(input_frames_dir, video_name)
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])

        if not frame_files:
            print(f"No hay frames para {video_name}")
            return False

        # Configurar video writer
        first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
        height, width = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Obtener todas las detecciones y grupos
        detections = db.query(FrameObjectDetection).filter_by(
            video_id=video.video_id
        ).all()

        groups = db.query(GroupDetection).filter_by(
            video_id=video.video_id
        ).all()

        # Organizar por frame
        detections_by_frame = defaultdict(dict)
        for det in detections:
            detections_by_frame[det.frame_number][det.track_id] = {
                'x1': det.x1, 'y1': det.y1,
                'x2': det.x2, 'y2': det.y2
            }

        groups_by_frame = defaultdict(list)
        for group in groups:
            # Obtener miembros del grupo
            members = db.query(GroupMember).filter_by(
                group_detection_id=group.id
            ).all()

            groups_by_frame[group.frame_number].append({
                'group_id': group.group_id,
                'features': {
                    'center_x': group.center_x,
                    'center_y': group.center_y,
                    'size': group.size,
                    'dispersion': group.dispersion,
                    'avg_velocity': group.avg_velocity
                },
                'members': [m.track_id for m in members]
            })

        # Generar paleta de colores
        max_group_id = max([g.group_id for g in groups]) + 1
        colors = generate_color_palette(max_group_id)

        # Procesar frames
        print(f"\nGenerando visualización de grupos para {video_name}...")
        for frame_file in tqdm(frame_files):
            try:
                frame_num = int(frame_file.split('_')[1].split('.')[0])
            except (IndexError, ValueError):
                continue

            frame_path = os.path.join(frames_dir, frame_file)
            frame = cv2.imread(frame_path)

            if frame is None:
                continue

            # Dibujar grupos
            if frame_num in groups_by_frame:
                detections_map = detections_by_frame.get(frame_num, {})

                for group_info in groups_by_frame[frame_num]:
                    color = colors[group_info['group_id'] % len(colors)]
                    frame = draw_group_visualization(
                        frame, group_info, detections_map, color
                    )

            # Información general del frame
            cv2.putText(frame, f"Frame: {frame_num}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            out.write(frame)

        out.release()
        print(f"\nVideo generado: {output_video_path}")
        return True

    except Exception as e:
        print(f"Error visualizando grupos en {video_name}: {str(e)}")
        return False
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualizar grupos detectados en videos"
    )
    parser.add_argument("--video_name", type=str, required=True,
                        help="Nombre del video a visualizar")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directorio con los frames")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directorio de salida para el video")
    parser.add_argument("--fps", type=int, default=30,
                        help="FPS del video de salida")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.video_name}_groups.mp4")

    visualize_groups_video(
        args.video_name,
        args.input_dir,
        output_path,
        args.fps
    )


if __name__ == "__main__":
    main()
