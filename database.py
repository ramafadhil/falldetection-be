# database.py
import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Format: postgresql://username:password@localhost:5432/nama_database
# Sesuaikan 'postgres' dan 'password_kamu' mengikut tetapan pgAdmin kamu
DATABASE_URL = "postgresql://postgres:ramafadhil123@localhost:5432/fall_detection_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()