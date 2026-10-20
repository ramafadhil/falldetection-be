# Elderly Fall Detection System 

Backend service dan engine AI deteksi jatuh menggunakan **FastAPI**, **PostgreSQL**, dan **YOLO11-pose**.

---

## Tech Stack
- **API Framework:** FastAPI & SQLAlchemy
- **Database:** PostgreSQL
- **AI Engine:** Ultralytics YOLO11 (`yolo11n-pose.pt` / `yolo11s-pose.pt`)
- **Computer Vision:** OpenCV & PyTorch (Auto CPU / CUDA fallback)

---

## Quickstart & Local Setup

### 1. Clone & Setup Virtual Environment
```bash
git clone https://github.com/ramafadhil/falldetection-be.git
cd falldetection-be

# Buat & aktifkan venv
python -m venv venv
.\venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Database
Pastikan PostgreSQL aktif, lalu sesuaikan URL koneksi di `database.py`:
```python
SQLALCHEMY_DATABASE_URL = "postgresql://<USER>:<PASSWORD>@localhost:5432/<DATABASE_NAME>"
```

---

## Cara Menjalankan

**1. Jalankan FastAPI Backend**
```bash
uvicorn main:app --reload
```
*API docs otomatis bisa diakses di `http://localhost:8000/docs`*

**2. Jalankan Script Deteksi Kamera (AI)**
Buka terminal baru (pastikan `venv` aktif), lalu jalankan:
```bash
python python_script/realtime_fall_detection.py
```
