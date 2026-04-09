import subprocess
from typing import List
from collections import defaultdict
import numpy as np

from src.database.connection import SessionLocal
from src.database.db_crud import VideoCRUD
from src.database.models import VideoMetadata, GroupDetection, GroupMember


class VideoFunctions:

    @staticmethod
    def convert_video_to_frames(video_folder: str, output_folder: str) -> None:
        print("\nProcesando video...")
        subprocess.run([
            "python",
            "src/features/video_to_frames_concurrent.py",
            "--video_dir", video_folder,
            "--output_folder", output_folder
        ])

    @staticmethod
    def detect_and_track(directories_selected, output_folder) -> None:
        subprocess.run([
            "python",
            "src/features/detect_tracking.py",
            "--input_dir", output_folder,
            "--folders", *directories_selected
        ])

    @staticmethod
    def analyze_groups(directories_selected, proximity_threshold, min_group_size) -> None:
        """Ejecuta el análisis de grupos para los videos seleccionados"""
        print("\nIniciando análisis de grupos...")
        for video_name in directories_selected:
            print(f"\nAnalizando: {video_name}")
            subprocess.run([
                "python",
                "src/features/group_analysis.py",
                "--video_name", video_name,
                "--proximity_threshold", str(proximity_threshold),
                "--min_group_size", str(min_group_size)
            ])

    @staticmethod
    def visualize_groups(directories_selected, input_folder, output_folder) -> None:
        """Genera videos con visualización de grupos"""
        print("\nGenerando visualizaciones de grupos...")
        output_dir = output_folder + "/group_videos"

        for video_name in directories_selected:
            print(f"\nVisualizando: {video_name}")
            subprocess.run([
                "python",
                "src/features/visualize_groups_pdf.py",
                "--video_name", video_name,
                "--input_dir", input_folder,
                "--output_dir", output_dir
            ])

    @staticmethod
    def analyze_groups_pdf(directories_selected, distance_threshold, min_frames) -> None:
        """Ejecuta el análisis de grupos usando el algoritmo del PDF"""
        print("\nIniciando análisis de grupos (algoritmo PDF)...")
        for video_name in directories_selected:
            print(f"\nAnalizando: {video_name}")
            subprocess.run([
                "python",
                "src/features/group_analysis_pdf.py",  # Nuevo archivo
                "--video_name", video_name,
                "--distance_threshold", str(distance_threshold),
                "--min_frames", str(min_frames)
            ])

    @staticmethod
    def show_group_statistics(directories_selected) -> None:
        """Muestra estadísticas detalladas de grupos"""
        session = SessionLocal()

        try:
            for video_name in directories_selected:
                print(f"\n{'='*60}")
                print(f"ESTADÍSTICAS DE GRUPOS - {video_name}")
                print(f"{'='*60}\n")

                # Obtener video
                video = session.query(VideoMetadata).filter_by(title=video_name).first()
                if not video:
                    print(f"Video {video_name} no encontrado")
                    continue

                # Obtener todos los grupos
                groups = session.query(GroupDetection).filter_by(
                    video_id=video.video_id
                ).all()

                if not groups:
                    print("No se encontraron grupos para este video")
                    continue

                # Agrupar por group_id
                groups_by_id = defaultdict(list)
                for group in groups:
                    groups_by_id[group.group_id].append(group)

                print(f"Total de grupos únicos: {len(groups_by_id)}\n")

                # Estadísticas por grupo
                for group_id, group_frames in groups_by_id.items():
                    print(f"--- Grupo {group_id} ---")

                    # Duración
                    frames = [g.frame_number for g in group_frames]
                    duration = len(frames)
                    print(f"  Duración: {duration} frames")
                    print(f"  Frames: {min(frames)} - {max(frames)}")

                    # Tamaño del grupo
                    sizes = [g.size for g in group_frames]
                    print(f"  Tamaño promedio: {np.mean(sizes):.1f} personas")
                    print(f"  Tamaño mín/máx: {min(sizes)}/{max(sizes)} personas")

                    # Detectar cambios de tamaño
                    size_changes = []
                    for i in range(1, len(group_frames)):
                        if group_frames[i].size != group_frames[i-1].size:
                            size_changes.append({
                                'frame': group_frames[i].frame_number,
                                'from': group_frames[i-1].size,
                                'to': group_frames[i].size
                            })

                    if size_changes:
                        print(f"  Cambios de tamaño: {len(size_changes)}")
                        for change in size_changes[:5]:  # Mostrar primeros 5
                            print(f"    Frame {change['frame']}: {change['from']} → {change['to']} personas")

                    # Velocidad
                    velocities = [g.avg_velocity for g in group_frames if g.avg_velocity]
                    if velocities:
                        print(f"  Velocidad promedio: {np.mean(velocities):.2f} px/frame")
                        print(f"  Velocidad máxima: {max(velocities):.2f} px/frame")

                    # Dispersión
                    dispersions = [g.dispersion for g in group_frames if g.dispersion]
                    if dispersions:
                        print(f"  Dispersión promedio: {np.mean(dispersions):.2f} px")
                        print(f"  Dispersión mín/máx: {min(dispersions):.2f}/{max(dispersions):.2f} px")

                    # Cohesión (velocidad estándar)
                    velocity_stds = [g.velocity_std for g in group_frames if g.velocity_std]
                    if velocity_stds:
                        avg_std = np.mean(velocity_stds)
                        print(f"  Cohesión (desv. velocidad): {avg_std:.2f}")
                        if avg_std < 10:
                            print(f"    → Grupo muy cohesionado")
                        elif avg_std < 30:
                            print(f"    → Grupo moderadamente cohesionado")
                        else:
                            print(f"    → Grupo poco cohesionado")

                    print()

                # Resumen general
                print(f"\n{'='*60}")
                print("RESUMEN GENERAL")
                print(f"{'='*60}")

                all_sizes = [g.size for g in groups]
                print(f"Tamaño promedio de todos los grupos: {np.mean(all_sizes):.1f}")

                all_velocities = [g.avg_velocity for g in groups if g.avg_velocity]
                if all_velocities:
                    print(f"Velocidad promedio general: {np.mean(all_velocities):.2f} px/frame")

                # Análisis temporal
                frames_with_groups = len(set([g.frame_number for g in groups]))
                total_frames = max([g.frame_number for g in groups]) - min([g.frame_number for g in groups]) + 1
                coverage = (frames_with_groups / total_frames) * 100 if total_frames > 0 else 0
                print(f"Cobertura de grupos: {coverage:.1f}% de los frames")

                print()

        except Exception as e:
            print(f"Error mostrando estadísticas: {e}")
            session.rollback()
        finally:
            session.close()

    @staticmethod
    def reconstruct_video(directories_selected, output_folder) -> None:
        subprocess.run([
            "python",
            "src/features/reconstruct_video.py",
            "--input_dir", output_folder + "/frames",
            "--folders", *directories_selected,
            "--output_folder", output_folder
        ])

    @staticmethod
    def get_videos_analyzed() -> List[VideoMetadata]:
        session = SessionLocal()
        try:
            videoCRUD = VideoCRUD(session)
            frames_selected = videoCRUD.get_all_videos_coverted()
            return frames_selected
        except Exception as e:
            print(f"Error al obtener frames analizados: {e}")
            session.rollback()
            return []
        finally:
            session.close()

    @staticmethod
    def get_frames_analyzed() -> List[VideoMetadata]:
        session = SessionLocal()
        try:
            videoCRUD = VideoCRUD(session)
            frames_selected = videoCRUD.get_all_videos_with_detections()
            return frames_selected
        except Exception as e:
            print(f"Error al obtener frames analizados: {e}")
            session.rollback()
            return []
        finally:
            session.close()

    @staticmethod
    def get_videos_with_groups() -> List[str]:
        """Obtiene lista de videos que tienen análisis de grupos"""
        session = SessionLocal()
        try:
            # Obtener videos únicos con grupos detectados
            videos = session.query(VideoMetadata.title).join(
                GroupDetection,
                VideoMetadata.video_id == GroupDetection.video_id
            ).distinct().all()

            return [video[0] for video in videos]
        except Exception as e:
            print(f"Error al obtener videos con grupos: {e}")
            session.rollback()
            return []
        finally:
            session.close()

    @staticmethod
    def command_run() -> None:
        """Ejecuta el comando de la CLI para iniciar el procesamiento de videos."""
        subprocess.run([
            "python",
            "src/features/terminal.py",
        ])
