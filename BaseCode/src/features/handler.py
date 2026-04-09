import os
import subprocess
import sys
import time

from src.features.video_functions import VideoFunctions
from src.menus.main_menu import MainMenu

class Handler:
    def __init__(self):
        self.video_folder = "/app/Videos/Pendientes"
        self.output_folder = os.getcwd()

    def main_menu(self, choice):
        if choice == '1':
            self.set_video_folder()
        elif choice == '2':
            self.set_output_folder()
        elif choice == '3':
            self.pipeline_options()
        elif choice == '0':
            self.terminal()
        elif choice == '4':
            print("Salir")
            sys.exit(1)
        else:
            print("\nOpcion invalida, intente de nuevo")

    def terminal(self):
        VideoFunctions.command_run()

    def set_video_folder(self):
        path = MainMenu.display_get_folder("\nIngresa la ruta de la carpeta con los videos", default=None)

        if os.path.exists(path):
            self.video_folder = path
            print(f"\n\tRuta seleccionada: {self.video_folder}")
        else:
            print("La ruta especificada no existe. Inténtelo de nuevo.")

    def set_output_folder(self):
        path = MainMenu.display_get_folder("\nIngresa la ruta de la carpeta con los videos", default=1)

        if path == "":
            self.output_folder = os.getcwd()
            print(f"\n\tRuta seleccionada: {self.output_folder}")
        elif os.path.exists(path):
            self.output_folder = path
            print(f"\n\tRuta seleccionada: {self.output_folder}")
        else:
            print("La ruta especificada no existe. Inténtelo de nuevo.")

    def pipeline_options(self):
        while True:
            choice = MainMenu.display_pipeline_options()

            if choice == '1':
                response = MainMenu.display_paths_confirmation(
                    video_folder=self.video_folder,
                    output_folder=self.output_folder
                )
                if response == 's':
                    start_time = time.time()
                    VideoFunctions.convert_video_to_frames(
                        self.video_folder,
                        self.output_folder
                    )

                    end_time = time.time()
                    execution_time = end_time - start_time
                    print(f"Conversión completada en: {self.format_time(execution_time)}")

            elif choice == '2':
                videos_converted = VideoFunctions.get_videos_analyzed()
                videos_selected = MainMenu.display_frame_folders(videos_converted)
                print(f"\n\tVideos seleccionados: {videos_selected}")
                start_time = time.time()
                VideoFunctions.detect_and_track(
                    videos_selected,
                    self.output_folder + "/frames"
                )
                end_time = time.time()
                execution_time = end_time - start_time
                print(f"Conversión completada en: {self.format_time(execution_time)}")

            elif choice == '3':
                print("\n=== ANÁLISIS DE GRUPOS (ALGORITMO PDF) ===")
                videos_with_detections = VideoFunctions.get_frames_analyzed()

                if not videos_with_detections:
                    print("No hay videos con detecciones. Ejecute primero la opción 2.")
                    continue

                videos_selected = MainMenu.display_frames_analysed(videos_with_detections)

                # Solicitar parámetros específicos del algoritmo del PDF
                print("\nParámetros de agrupamiento (basados en el PDF):")
                distance_threshold = input("Distancia máxima de proximidad en píxeles (default: 100): ").strip()
                distance_threshold = float(distance_threshold) if distance_threshold else 100.0

                min_frames = input("Mínimo número de frames para formar grupo (default: 15): ").strip()
                min_frames = int(min_frames) if min_frames else 15

                start_time = time.time()
                VideoFunctions.analyze_groups_pdf(
                    videos_selected,
                    distance_threshold,
                    min_frames
                )
                end_time = time.time()
                execution_time = end_time - start_time
                print(f"Conversión completada en: {self.format_time(execution_time)}")

            elif choice == '4':
                print("\n=== ESTADÍSTICAS DE GRUPOS ===")
                videos_with_groups = VideoFunctions.get_videos_with_groups()

                if not videos_with_groups:
                    print("No hay videos con análisis de grupos. Ejecute primero la opción 3.")
                    continue

                videos_selected = MainMenu.display_videos_with_groups(videos_with_groups)
                VideoFunctions.show_group_statistics(videos_selected)

            elif choice == '5':
                print("\n=== VISUALIZACIÓN DE GRUPOS ===")
                videos_with_groups = VideoFunctions.get_videos_with_groups()

                if not videos_with_groups:
                    print("No hay videos con análisis de grupos. Ejecute primero la opción 3.")
                    continue

                videos_selected = MainMenu.display_videos_with_groups(videos_with_groups)
                VideoFunctions.visualize_groups(
                    videos_selected,
                    self.output_folder + "/frames",
                    self.output_folder
                )

            elif choice == '6':
                frames_analyzed = VideoFunctions.get_frames_analyzed()
                frames_selected = MainMenu.display_frames_analysed(frames_analyzed)
                VideoFunctions.reconstruct_video(frames_selected, self.output_folder)

            elif choice == '7':
                print("Regresar al menu anterior")
                break
            else:
                print("\nOpcion invalida, intente de nuevo")

    @staticmethod
    def format_time(seconds):
        """Convierte segundos a formato más legible"""
        if seconds < 60:
            return f"{seconds:.2f} segundos"
        elif seconds < 3600:
            minutes = seconds // 60
            seconds = seconds % 60
            return f"{int(minutes)} minutos y {seconds:.2f} segundos"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            seconds = seconds % 60
            return f"{int(hours)} horas, {int(minutes)} minutos y {seconds:.2f} segundos"
