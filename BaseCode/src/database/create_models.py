from connection import engine
from models import Base
from sqlalchemy import inspect
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_tables_exist():
    """Verifica si todas las tablas definidas en los modelos ya existen en la BD"""
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    required_tables = list(Base.metadata.tables.keys())

    logger.info(f"Tablas requeridas: {required_tables}")
    logger.info(f"Tablas existentes: {existing_tables}")

    missing_tables = [table for table in required_tables if table not in existing_tables]

    if missing_tables:
        logger.warning(f"Faltan {len(missing_tables)} tablas: {missing_tables}")
        return False
    logger.info("Todas las tablas ya existen en la base de datos")
    return True

def create_tables_if_needed():
    """Crea las tablas solo si no existen"""
    if not check_tables_exist():
        logger.info("Creando tablas...")
        Base.metadata.create_all(bind=engine)
        logger.info("Tablas creadas exitosamente")
        return True
    return False

if __name__ == "__main__":
    create_tables_if_needed()
