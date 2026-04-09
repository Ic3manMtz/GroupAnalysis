import numpy as np
import argparse
from collections import defaultdict, deque
from sqlalchemy.orm import sessionmaker
from src.database.connection import SessionLocal
from src.database.models import FrameObjectDetection, VideoMetadata, GroupDetection, GroupMember
from tqdm import tqdm
import math


class GroupTracker:
    def __init__(self, distance_threshold=100, min_frames=15):
        """
        Args:
            distance_threshold: Distancia máxima en píxeles para considerar proximidad
            min_frames: Número mínimo de frames para formar un grupo (basado en el PDF)
        """
        self.distance_threshold = distance_threshold
        self.min_frames = min_frames
        self.next_group_id = 1


        # Estructuras para tracking (como en el PDF)
        self.pair_frames = defaultdict(int)  # (id1, id2) -> frames consecutivos cerca
        self.group_assignments = {}  # track_id -> group_id
        self.active_groups = defaultdict(set)  # group_id -> set de track_ids
        self.group_history = defaultdict(lambda: defaultdict(int))  # group_id -> {track_id: frames}


    def calculate_distance(self, pos1, pos2):
        """Calcula distancia L2 entre dos puntos (x,y)"""
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def calculate_centroid(self, box):
        """Calcula el centroide de un bounding box"""
        return ((box['x1'] + box['x2']) / 2, (box['y1'] + box['y2']) / 2)

    def update(self, detections):
        """
        Actualiza el estado de grupos basado en las detecciones del frame actual
        Implementa el algoritmo descrito en el PDF
        """
        # Paso 1: Calcular distancias entre todos los pares
        current_centroids = {}
        for det in detections:
            current_centroids[det['track_id']] = self.calculate_centroid(det)

        # Encontrar pares cercanos en este frame
        current_pairs = set()
        track_ids = list(current_centroids.keys())

        for i in range(len(track_ids)):
            for j in range(i + 1, len(track_ids)):
                id1, id2 = track_ids[i], track_ids[j]
                distance = self.calculate_distance(
                    current_centroids[id1],
                    current_centroids[id2]
                )

                if distance < self.distance_threshold:
                    pair = tuple(sorted([id1, id2]))
                    current_pairs.add(pair)


        # Paso 2: Actualizar contadores de frames para cada par
        # Incrementar contadores para pares que están cerca
        for pair in current_pairs:
            self.pair_frames[pair] += 1

        # Resetear contadores para pares que ya no están cerca
        all_pairs = set(self.pair_frames.keys())
        for pair in all_pairs - current_pairs:
            self.pair_frames[pair] = 0

        # Paso 3: Identificar pares que han estado cerca suficiente tiempo
        confirmed_pairs = set()
        for pair, frames in self.pair_frames.items():
            if frames >= self.min_frames:
                confirmed_pairs.add(pair)


        # Paso 4: Construir grupos a partir de pares confirmados
        # Crear grafo de conexiones
        graph = defaultdict(set)
        for id1, id2 in confirmed_pairs:
            graph[id1].add(id2)
            graph[id2].add(id1)

        # Encontrar componentes conectados (grupos)
        visited = set()
        current_groups = []

        for node in graph:
            if node not in visited:
                # BFS para encontrar componente conectado
                stack = [node]
                component = set()
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.add(current)
                        stack.extend(graph[current] - visited)
                current_groups.append(component)

        # Paso 5: Asignar/actualizar group_ids
        new_group_assignments = {}

        for group_members in current_groups:
            # Buscar group_id existente para algún miembro
            existing_group_id = None
            for member in group_members:
                if member in self.group_assignments:
                    existing_group_id = self.group_assignments[member]
                    break

            # Asignar nuevo group_id si no existe
            if existing_group_id is None:
                existing_group_id = self.next_group_id
                self.next_group_id += 1

            # Asignar group_id a todos los miembros
            for member in group_members:
                new_group_assignments[member] = existing_group_id
                self.group_history[existing_group_id][member] += 1

        # Actualizar asignaciones globales
        self.group_assignments = new_group_assignments

        # Actualizar grupos activos
        self.active_groups.clear()
        for track_id, group_id in self.group_assignments.items():
            self.active_groups[group_id].add(track_id)

        # Paso 6: Preparar resultado
        groups_output = []
        for group_id, members in self.active_groups.items():
            if len(members) >= 2:  # Solo grupos con 2+ personas
                groups_output.append({
                    'group_id': group_id,
                    'members': list(members),
                    'size': len(members),
                    'type': 'couple' if len(members) == 2 else 'group'
                })

        # Identificar individuos (personas no en grupos)
        individuals = []
        all_track_ids = set(track_ids)
        grouped_track_ids = set(self.group_assignments.keys())
        individual_track_ids = all_track_ids - grouped_track_ids

        for track_id in individual_track_ids:
            individuals.append(track_id)

        return {
            'groups': groups_output,
            'individuals': individuals,
            'all_detections': detections
        }

def analyze_video_groups(video_name, distance_threshold=100, min_frames=15):
    """Analiza grupos en un video completo usando el algoritmo del PDF"""
    db = SessionLocal()
    tracker = GroupTracker(distance_threshold, min_frames)

    try:
        # Obtener video
        video = db.query(VideoMetadata).filter_by(title=video_name).first()
        if not video:
            print(f"Video {video_name} no encontrado")
            return

        # Obtener todas las detecciones ordenadas por frame
        detections = db.query(FrameObjectDetection).filter_by(
            video_id=video.video_id
        ).order_by(FrameObjectDetection.frame_number).all()

        if not detections:
            print(f"No hay detecciones para el video {video_name}")
            return

        # Organizar detecciones por frame
        frames_data = defaultdict(list)
        for det in detections:
            frame_data = {
                'track_id': det.track_id,
                'x1': det.x1, 'y1': det.y1,
                'x2': det.x2, 'y2': det.y2
            }
            frames_data[det.frame_number].append(frame_data)

        # Procesar cada frame
        all_groups_data = []
        print(f"Analizando grupos en {video_name}...")

        for frame_num in tqdm(sorted(frames_data.keys())):
            frame_detections = frames_data[frame_num]
            result = tracker.update(frame_detections)

            # Guardar grupos en base de datos
            for group_info in result['groups']:
                # Calcular centro del grupo
                centers = []
                for track_id in group_info['members']:
                    det = next((d for d in frame_detections if d['track_id'] == track_id), None)
                    if det:
                        center_x = (det['x1'] + det['x2']) / 2
                        center_y = (det['y1'] + det['y2']) / 2
                        centers.append((center_x, center_y))

                if centers:
                    center_x = np.mean([c[0] for c in centers])
                    center_y = np.mean([c[1] for c in centers])

                    # Calcular dispersión
                    dispersion = np.mean([
                        np.linalg.norm(np.array([center_x, center_y]) - np.array([c[0], c[1]]))
                        for c in centers
                    ])
                else:
                    center_x, center_y, dispersion = 0, 0, 0

                # Guardar detección de grupo
                group_detection = GroupDetection(
                    video_id=video.video_id,
                    frame_number=frame_num,
                    group_id=group_info['group_id'],
                    center_x=float(center_x),
                    center_y=float(center_y),
                    size=group_info['size'],
                    dispersion=float(dispersion),
                    avg_velocity=0.0,  # Podrías calcular esto con tracking entre frames
                    velocity_std=0.0
                )
                db.add(group_detection)
                db.flush()

                # Guardar miembros del grupo
                for track_id in group_info['members']:
                    group_member = GroupMember(
                        group_detection_id=group_detection.id,
                        track_id=track_id
                    )
                    db.add(group_member)

            all_groups_data.extend(result['groups'])

        db.commit()
        print(f"Procesados {len(frames_data)} frames")
        print(f"Encontrados {len(set(g['group_id'] for g in all_groups_data))} grupos únicos")

    except Exception as e:
        print(f"Error analizando grupos en {video_name}: {str(e)}")
        db.rollback()
        import traceback
        traceback.print_exc()

    finally:
        db.close()



def main():
    parser = argparse.ArgumentParser(description="Análisis de grupos basado en el algoritmo del PDF")

    parser.add_argument("--video_name", type=str, required=True,
                        help="Nombre del video a analizar")

    parser.add_argument("--distance_threshold", type=float, default=100.0,
                        help="Distancia máxima para considerar proximidad (píxeles)")

    parser.add_argument("--min_frames", type=int, default=15,
                        help="Mínimo número de frames para formar un grupo")

    args = parser.parse_args()

    analyze_video_groups(
        args.video_name,
        args.distance_threshold,
        args.min_frames
    )


if __name__ == "__main__":
    main()
