
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, JSON, String

from database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    emotion = Column(String(20), nullable=False)
    score = Column(Float, nullable=False)
    probabilities = Column(JSON, nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
