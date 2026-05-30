import os
from time import sleep
from typing import List, Tuple, Optional
from colorama import init, Fore, Style


class MainMenu:
    # Definimos las opciones como variables de clase estáticas
    MAIN_OPTIONS: List[Tuple[str, str]] = [
        ("1", "Seleccionar carpeta de videos default='/app/Videos/Pendientes'"),
        ("2", "Seleccionar carpeta de salida default='/app'"),
        ("3", "Pipeline de análisis de videos"),
        ("4", "Salir")
    ]

    PIPELINE_OPTIONS: List[Tuple[str, str]] = [
        ("1", "Convertir videos a frames"),
        ("2", "Detección de objetos y seguimiento"),
        ("3", "Análisis de grupos de personas"),
        ("4", "Estadísticas de grupos"),
        ("5", "Visualización de grupos en videos"),
        ("6", "Visualización de videos con detecciones individuales"),
        ("7", "Volver al menú principal")
    ]

    @staticmethod
    def display_main_menu(converted_count: Optional[int] = None) -> str:
        print("\n" + "=" * 40)
        print(" " * 10 + "ANÁLISIS DE VIDEOS".center(20))
        print("=" * 40 + "\n")

        options = MainMenu.MAIN_OPTIONS.copy()

        for num, text in options:
            #print(f" {num}┃ {text}")
            text = MainMenu.highlight_text(text)
            print(f" {num:>3}┃ {text}")

        print("\n" + "-" * 40)
        return input(" ➤ Seleccione una opción: ")

    @staticmethod
    def display_get_folder(prompt, default=1) -> str:
        if default:
            prompt += f" (Enter para usar la ruta actual): "
        else:
            prompt += ": "
        return input(prompt)

    @staticmethod
    def display_pipeline_options():
        print("\n" + "="*40)
        print(" " * 10 + "PIPELINE DE ANÁLISIS".center(20))
        print("="*40 + "\n")

        for num, text in MainMenu.PIPELINE_OPTIONS:
            print(f" {num}┃ {text}")

        print("\n" + "-"*40)
        return input(" ➤ Seleccione una opción: ")

    @staticmethod
    def highlight_text(text: str) -> str:

        if "'" not in text:
            return text

        before_quote, remaining = text.split("'", 1)

        # Split the remaining part to get the text to highlight
        if "'" in remaining:
            highlight_content, after_quote = remaining.split("'", 1)
        else:
            highlight_content = remaining
            after_quote = ""

        # Build the final result
        return f"{before_quote}'{Fore.YELLOW}{highlight_content}{Style.RESET_ALL}'{after_quote}"



    @staticmethod
    def display_paths_confirmation(video_folder: str, output_folder: str) -> str:
        print(f"\nCarpeta de videos: {video_folder}")
        print(f"Carpeta de salida: {output_folder}")
        return input("¿Son correctas estas rutas? (s/n): ").lower()

    @staticmethod
    def display_frame_folders(videos_converted):
        print("\n" + "=" * 40)
        print(" " * 5 + "VIDEOS CONVERTIDOS A FRAMES".center(20))
        print("=" * 40 + "\n")

        print("Seleccione los videos que desea usar (separe los números con comas):")
        for idx, video in enumerate(videos_converted, 1):
            print(f"  {idx}. {video}")

        selection = input(
            "\nIngrese los números de las carpetas, separados por comas (Enter para seleccionar todas): ").strip()

        if not selection:
            # Si el usuario presiona Enter, selecciona todas las carpetas
            videos_selected = videos_converted
        else:
            # Procesar la entrada del usuario
            indices = [int(i.strip()) for i in selection.split(",") if i.strip().isdigit()]
            # Seleccionar los videos correspondientes (usando videos_converted en lugar de video)
            videos_selected = [videos_converted[i - 1] for i in indices if 1 <= i <= len(videos_converted)]

        return videos_selected

    @staticmethod
    def display_frames_analysed(frames_analyzed):
        print("\n" + "=" * 40)
        print(" " * 5 + "VIDEOS CON FRAMES ANALIZADOS".center(20))
        print("=" * 40 + "\n")

        print("Seleccione los videos que desea usar (separe los números con comas):")
        for idx, frame in enumerate(frames_analyzed, 1):
            print(f"  {idx}. {frame}")

        selection = input(
            "\nIngrese los números de las carpetas, separados por comas (Enter para seleccionar todas): ").strip()

        if not selection:
            # Si el usuario presiona Enter, selecciona todas las carpetas
            frames_selected = frames_analyzed
        else:
            indices = [int(i.strip()) for i in selection.split(",") if i.strip().isdigit()]
            frames_selected = [frames_analyzed[i - 1] for i in indices if 1 <= i <= len(frames_analyzed)]

        return frames_selected

    @staticmethod
    def display_videos_with_groups(videos_with_groups):
        """Muestra menú de videos con análisis de grupos"""
        print("\n" + "=" * 40)
        print(" " * 5 + "VIDEOS CON ANÁLISIS DE GRUPOS".center(20))
        print("=" * 40 + "\n")

        print("Seleccione los videos que desea usar (separe los números con comas):")
        for idx, video in enumerate(videos_with_groups, 1):
            print(f"  {idx}. {video}")

        selection = input(
            "\nIngrese los números de los videos, separados por comas (Enter para todos): ").strip()

        if not selection:
            videos_selected = videos_with_groups
        else:
            indices = [int(i.strip()) for i in selection.split(",") if i.strip().isdigit()]
            videos_selected = [videos_with_groups[i - 1] for i in indices if 1 <= i <= len(videos_with_groups)]

        return videos_selected
