import os
import time
import subprocess
from fastapi import FastAPI, Depends, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from typing import List

import models, schemas, database

# Buat tabel otomatis jika belum ada
models.Base.metadata.create_all(bind=database.engine)

app = FastAPI(title="Fall Detection API")

# Setup CORS agar Next.js (port 3000) bisa akses
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount folder storage untuk akses gambar snapshot dari URL HTTP
os.makedirs("storage/thumbnails", exist_ok=True)
app.mount("/storage", StaticFiles(directory="storage"), name="storage")

detection_process = None

@app.get("/")
def root():
    return {"status": "ok", "message": "Fall Detection API is running"}

# --- ENDPOINTS INCIDENT ALERTS ---

@app.get("/api/incident-alerts", response_model=List[schemas.AlertResponse])
def get_alerts(db: Session = Depends(database.get_db)):
    alerts = db.query(models.IncidentAlert).order_by(models.IncidentAlert.event_timestamp.desc()).all()
    return alerts

@app.post("/api/incident-alerts")
async def create_alert(
    device_id: int = Form(...),
    confidence: float = Form(...),
    thumbnail: UploadFile = File(None),
    db: Session = Depends(database.get_db)
):
    file_path = None
    if thumbnail:
        # Beri nama unik pakai timestamp agar tidak berbenturan
        timestamp_str = int(time.time())
        filename = f"thumb_dev{device_id}_{timestamp_str}.jpg"
        file_path = f"storage/thumbnails/{filename}"
        
        with open(file_path, "wb") as f:
            f.write(await thumbnail.read())

    new_alert = models.IncidentAlert(
        device_id=device_id,
        confidence=confidence,
        thumbnail_path=file_path
    )
    db.add(new_alert)
    db.commit()
    db.refresh(new_alert)
    return new_alert

@app.delete("/api/incident-alerts/{alert_id}")
def delete_alert(alert_id: int, db: Session = Depends(database.get_db)):
    alert = db.query(models.IncidentAlert).filter(models.IncidentAlert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    if alert.thumbnail_path and os.path.exists(alert.thumbnail_path):
        os.remove(alert.thumbnail_path)
        
    db.delete(alert)
    db.commit()
    return {"message": "Alert deleted successfully"}

# --- ENDPOINTS CONTROLLER DETECTION ENGINE ---

@app.post("/api/detection/start")
def start_detection():
    global detection_process
    if detection_process and detection_process.poll() is None:
        return {"status": "already_running", "message": "Detection script is already running"}

    # Jalankan script realtime_fall_detection.py dengan venv python
    script_path = os.path.join("python_script", "realtime_fall_detection.py")
    detection_process = subprocess.Popen(["python", script_path])
    return {"status": "started", "message": "Detection script started"}

@app.post("/api/detection/stop")
def stop_detection():
    global detection_process
    if detection_process and detection_process.poll() is None:
        detection_process.terminate()
        detection_process = None
        return {"status": "stopped", "message": "Detection script stopped"}
    return {"status": "not_running", "message": "Detection script is not running"}

@app.get("/api/detection/status")
def detection_status():
    global detection_process
    is_running = detection_process is not None and detection_process.poll() is None
    return {"is_running": is_running}

# --- AUTO SEED DEFAULT DEVICE ---
def seed_default_device():
    db: Session = database.SessionLocal()
    try:
        device = db.query(models.Device).filter(models.Device.id == 1).first()
        if not device:
            default_device = models.Device(
                id=1,
                name="Webcam Utama"
            )
            db.add(default_device)
            db.commit()
            print("[DB SEED] Device ID #1 berhasil dibuat.")
    except Exception as e:
        print(f"[DB SEED ERROR] Gagal buat device: {e}")
        db.rollback()
    finally:
        db.close()

seed_default_device()