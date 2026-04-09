from sqlalchemy import select, distinct
from sqlalchemy.orm import Session
from src.database.models import VideoMetadata, FrameObjectDetection
from typing import List, Optional

class VideoCRUD:
    def __init__(self, db: Session):
        self.db = db

    def create_video(self, title: str, duration: float, size: float) -> VideoMetadata:
        video = VideoMetadata(
            title=title,
            duration=duration,
            size=size
        )
        self.db.add(video)
        self.db.commit()
        self.db.refresh(video)
        return video

    def get_video_by_id(self, video_id: int) -> Optional[VideoMetadata]:
        return self.db.query(VideoMetadata).filter(VideoMetadata.video_id == video_id).first()

    def get_video_by_title(self, title: str) -> Optional[VideoMetadata]:
        return self.db.query(VideoMetadata).filter(VideoMetadata.title == title).first()

    def get_all_videos_coverted(self) -> List[VideoMetadata]:
        try:
            return [
                result[0] for result in
                self.db.query(VideoMetadata.title).outerjoin(
                    FrameObjectDetection,
                    VideoMetadata.video_id == FrameObjectDetection.video_id
                ).filter(
                    FrameObjectDetection.video_id.is_(None)
                ).distinct().all()
            ]
        except Exception as e:
            self.db.rollback()
            print(f"Error al obtener videos con detecciones: {e}")
            return []

    def get_all_videos_with_detections(self) -> List[VideoMetadata]:
        try:
            return [
            result[0] for result in
            self.db.query(VideoMetadata.title)
            .join(FrameObjectDetection)
            .filter(VideoMetadata.video_id == FrameObjectDetection.video_id)
            .distinct()
            .all()
            ]
        except Exception as e:
            self.db.rollback()
            print(f"Error al obtener videos con detecciones: {e}")
            return []

    def update_video(self, video_id: int, **kwargs) -> Optional[VideoMetadata]:
        video = self.get_video_by_id(video_id)
        if video:
            for key, value in kwargs.items():
                setattr(video, key, value)
            self.db.commit()
            self.db.refresh(video)
        return video

    def delete_video(self, video_id: int) -> bool:
        video = self.get_video_by_id(video_id)
        if video:
            self.db.delete(video)
            self.db.commit()
            return True
        return False

    def create_Frame(self, video_title: str, frame_number: int, track_id: int,
                     x1: float, y1: float, x2: float, y2: float) -> FrameObjectDetection:

        # Primero busca o crea el video
        video = self.db.query(VideoMetadata).filter_by(video_title=video_title).first()

        if not video:
            video = VideoMetadata(video_title=video_title)
            self.db.add(video)
            self.db.flush()  # Para obtener el video_id antes del commit

        # Crea la detección
        frame = FrameObjectDetection(
            video_id=video.video_id,
            frame_number=frame_number,
            track_id=track_id,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2
        )

        self.db.add(frame)
        self.db.commit()
        self.db.refresh(frame)
        return frame
