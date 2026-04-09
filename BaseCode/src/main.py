import sys
import subprocess
from time import sleep
from menus.main_menu import MainMenu
from features.handler import Handler

def main(handler):
    while True:
        choice = MainMenu.display_main_menu()
        handler.main_menu(choice)

def correr_pruebas():
    print("Probando GPU...")
    subprocess.run([sys.executable, "tests/gpu_test.py"])

    print("\nProbando instalación de torch...")
    subprocess.run([sys.executable, "tests/torch_test.py"])

    print("\nProbando modelo de YOLO...")
    subprocess.run([sys.executable, "tests/yolo_test.py"])

if __name__ == "__main__":
    if(len(sys.argv) > 1):
        if sys.argv[1] == "start":
            correr_pruebas()

    handler = Handler()
    main(handler)
