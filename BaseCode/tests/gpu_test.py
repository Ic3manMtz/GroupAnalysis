import torch

if torch.cuda.is_available():
    print("\t¡CUDA funciona correctamente!")
    print(f"\tDispositivo: {torch.cuda.get_device_name(0)}")
else:
    print("\tERROR: No se detectó GPU/CUDA")
    print("\tPosibles causas:")
    print("\t1. Fallo en el mapeo de dispositivos")
    print("\t2. Versiones incompatibles de drivers/CUDA")
    print("\t3. Falta de librerías en el contenedor")
