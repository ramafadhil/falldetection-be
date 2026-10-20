from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

class Device(Base):
    __tablename__ = "devices"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    alerts = relationship("IncidentAlert", back_populates="device")

class IncidentAlert(Base):
    __tablename__ = "incident_alerts"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(Integer, ForeignKey("devices.id"))
    confidence = Column(Float, nullable=False)
    event_timestamp = Column(DateTime, default=datetime.utcnow)
    thumbnail_path = Column(Text, nullable=True)

    device = relationship("Device", back_populates="alerts")