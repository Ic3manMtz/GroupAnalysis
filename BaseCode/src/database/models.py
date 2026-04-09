from sqlalchemy import Column, BigInteger, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class VideoMetadata(Base):
    __tablename__ = 'video_metadata'

    video_id = Column(BigInteger, primary_key=True)
    title = Column(String(255), nullable=False)
    duration = Column(Float, nullable=False)
    size = Column(Float, nullable=False)

    # Relación con FrameObjectDetection (usando el nombre correcto)
    object_detections = relationship(
        "FrameObjectDetection",
        back_populates="video",
        cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<VideoMetadata(video_id={self.video_id}, title='{self.title}')>"


class FrameObjectDetection(Base):
    __tablename__ = 'frame_object_detections'

    id = Column(BigInteger, primary_key=True)
    video_id = Column(BigInteger, ForeignKey('video_metadata.video_id'), nullable=False)
    frame_number = Column(BigInteger, nullable=False)
    track_id = Column(BigInteger, nullable=False)
    x1 = Column(Float, nullable=False)
    y1 = Column(Float, nullable=False)
    x2 = Column(Float, nullable=False)
    y2 = Column(Float, nullable=False)

    # Relación con VideoMetadata (nombre consistente)
    video = relationship("VideoMetadata", back_populates="object_detections")

    def __repr__(self):
        return f"<FrameObjectDetection(id={self.id}, frame={self.frame_number}, track={self.track_id})>"

# Agregar estos modelos a tu archivo models.py existente

class GroupDetection(Base):
    __tablename__ = 'group_detections'

    id = Column(BigInteger, primary_key=True)
    video_id = Column(BigInteger, ForeignKey('video_metadata.video_id'), nullable=False)
    frame_number = Column(BigInteger, nullable=False)
    group_id = Column(BigInteger, nullable=False)  # ID consistente del grupo a través de frames

    # Características del grupo
    center_x = Column(Float, nullable=False)
    center_y = Column(Float, nullable=False)
    size = Column(BigInteger, nullable=False)  # Número de personas en el grupo
    dispersion = Column(Float)  # Dispersión espacial del grupo
    avg_velocity = Column(Float)  # Velocidad promedio del grupo
    velocity_std = Column(Float)  # Desviación estándar de velocidades

    # Relaciones
    video = relationship("VideoMetadata", backref="group_detections")
    members = relationship("GroupMember", back_populates="group", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<GroupDetection(id={self.id}, group_id={self.group_id}, size={self.size})>"


class GroupMember(Base):
    __tablename__ = 'group_members'

    id = Column(BigInteger, primary_key=True)
    group_detection_id = Column(BigInteger, ForeignKey('group_detections.id'), nullable=False)
    track_id = Column(BigInteger, nullable=False)

    # Relación
    group = relationship("GroupDetection", back_populates="members")

    def __repr__(self):
        return f"<GroupMember(track_id={self.track_id})>"


class GroupStatistics(Base):
    __tablename__ = 'group_statistics'

    id = Column(BigInteger, primary_key=True)
    video_id = Column(BigInteger, ForeignKey('video_metadata.video_id'), nullable=False)
    group_id = Column(BigInteger, nullable=False)

    # Estadísticas agregadas del grupo
    first_frame = Column(BigInteger)
    last_frame = Column(BigInteger)
    duration_frames = Column(BigInteger)
    avg_size = Column(Float)
    min_size = Column(BigInteger)
    max_size = Column(BigInteger)
    avg_velocity = Column(Float)
    max_velocity = Column(Float)
    avg_dispersion = Column(Float)

    # Cambios en el grupo
    size_changes = Column(JSON)  # Lista de cambios de tamaño con frames

    video = relationship("VideoMetadata", backref="group_statistics")

    def __repr__(self):
        return f"<GroupStatistics(video_id={self.video_id}, group_id={self.group_id})>"
