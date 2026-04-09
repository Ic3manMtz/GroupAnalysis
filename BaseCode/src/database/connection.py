from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os


# Cargar variables de entorno
load_dotenv()

# Obtener connection string con validación
DATABASE_URL = os.getenv("DB_CONNECTION_STRING")

if not DATABASE_URL:
    raise ValueError(
        "No se encontró DB_CONNECTION_STRING en las variables de entorno. "
        "Por favor crea un archivo .env con esta variable."
    )

# Configurar el motor de SQLAlchemy
try:
    if 'engine' in globals():
        engine.dispose()

    engine = create_engine(
        DATABASE_URL,
        pool_size=50, # Tamaño del pool de conexiones
        max_overflow=20,  # Permitir hasta 20 conexiones adicionales
        pool_pre_ping=True,
        pool_recycle=3600  # Reciclar conexiones cada 1 hora
    )

    # Verificar la conexión
    with engine.connect() as test_conn:
        print("Conexión a PostgreSQL establecida correctamente")

except Exception as e:
    print("Error al conectar a la base de datos:")
    print(f"URL usada: {DATABASE_URL}")
    print(f"Error detallado: {e}")
    raise

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
