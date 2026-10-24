import time
import math
from collections import deque
import os

import cv2
import requests
from ultralytics import YOLO
import torch

# =========================
# KONFIGURASI
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Gunakan 0 untuk webcam, atau path file video tes
VIDEO_SOURCE = os.path.join(BASE_DIR, "test_fall.mp4") 
CONF_THRESHOLD = 0.25                # Minimum confidence deteksi orang

# Threshold Rule-based
ASPECT_RATIO_THRESHOLD = 0.55        # width/height bbox > 0.55 -> indikasi rebah
TORSO_ANGLE_THRESHOLD = 25           # derajat dari vertikal -> indikasi rebah
VERTICAL_VELOCITY_THRESHOLD = 20     # px/frame turun cepat -> indikasi jatuh

INACTIVITY_SECONDS = 1.0             # Durasi diam (detik) untuk konfirmasi jatuh
POSTURE_WINDOW = 5                   # Jumlah frame window untuk voting mayoritas
POSTURE_VOTE_RATIO = 0.4             # Minimal 40% TRUE dalam window agar lebih sensitif

ALERT_API_URL = "http://127.0.0.1:8000/api/incident-alerts"

# Index keypoint COCO Pose
NOSE, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 0, 5, 6, 11, 12

torch.set_num_threads(4)


def calc_torso_angle(shoulder_mid, hip_mid):
    """Menghitung sudut kemiringan torso terhadap sumbu vertikal."""
    dx = hip_mid[0] - shoulder_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]
    return math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))


def calc_aspect_ratio(box):
    """Menghitung rasio lebar dibanding tinggi bounding box."""
    x1, y1, x2, y2 = box
    w, h = (x2 - x1), (y2 - y1)
    return w / (h + 1e-6)


def midpoint(p1, p2):
    """Menghitung titik tengah antara dua titik."""
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


class PersonTracker:
    """Menyimpan histori posisi tiap orang (per track_id)."""

    def __init__(self, history_len=10):
        self.hip_y_history = deque(maxlen=history_len)
        self.posture_flags = deque(maxlen=POSTURE_WINDOW)
        self.fall_suspected_since = None
        self.last_hip_pos = None
        self.alert_sent = False

    def update_posture_flag(self, is_fall_posture):
        """Voting mayoritas posture flag per-frame agar tahan noise."""
        self.posture_flags.append(is_fall_posture)
        if len(self.posture_flags) < 2:
            return False
        true_ratio = sum(self.posture_flags) / len(self.posture_flags)
        return true_ratio >= POSTURE_VOTE_RATIO

    def update_hip(self, hip_mid):
        self.hip_y_history.append(hip_mid[1])
        self.last_hip_pos = hip_mid

    def vertical_velocity(self):
        if len(self.hip_y_history) < 2:
            return 0.0
        return self.hip_y_history[-1] - self.hip_y_history[0]

    def is_inactive(self, current_hip, movement_threshold=40):
        if self.last_hip_pos is None or current_hip is None:
            return True  # Anggap diam jika keypoint hilang saat tergeletak
        dist = math.dist(current_hip, self.last_hip_pos)
        return dist < movement_threshold


def send_alert(frame, track_id, confidence):
    """Kirim event alert + snapshot thumbnail ke backend FastAPI."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    filename = f"temp_alert_{track_id}_{int(time.time())}.jpg"
    
    cv2.imwrite(filename, frame)
    payload = {
        "device_id": 1,
        "confidence": round(confidence, 2)
    }

    try:
        with open(filename, "rb") as img_file:
            response = requests.post(
                ALERT_API_URL,
                data=payload,
                files={"thumbnail": img_file},
                timeout=3,
            )
        
        if response.status_code == 200:
            print(f"[ALERT SENT] Success sending alert for Track ID #{track_id}")
        else:
            print(f"[WARNING] Server error {response.status_code}: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"[WARNING] Gagal koneksi ke FastAPI: {e}")
    
    finally:
        if os.path.exists(filename):
            os.remove(filename)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = "yolo11n-pose.pt" if device == "cpu" else "yolo11s-pose.pt"
    print(f"[INFO] Device detected: {device} | Using model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH).to(device)

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    trackers = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Video playback completed or unable to read frame.")
            break

        results = model.track(
            frame,
            imgsz=480, 
            persist=True, 
            conf=CONF_THRESHOLD, 
            tracker="bytetrack.yaml", 
            verbose=False
        )
        annotated_frame = results[0].plot()

        if results[0].boxes.id is not None and results[0].keypoints is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            keypoints_xy = results[0].keypoints.xy.cpu().numpy()
            keypoints_data = results[0].keypoints.data.cpu().numpy()

            for i, (box, track_id, kpts) in enumerate(zip(boxes, track_ids, keypoints_xy)):
                if track_id not in trackers:
                    trackers[track_id] = PersonTracker()
                tracker = trackers[track_id]

                raw_posture_flag = False
                person_kpts = keypoints_data[i]
                aspect_ratio = calc_aspect_ratio(box)

                # Fleksibel: Paling tidak satu pinggul > 0.30
                l_hip_conf = person_kpts[L_HIP][2] if len(person_kpts) > L_HIP else 0
                r_hip_conf = person_kpts[R_HIP][2] if len(person_kpts) > R_HIP else 0
                has_valid_hips = (l_hip_conf > 0.30) or (r_hip_conf > 0.30)

                hip_mid = None
                if has_valid_hips:
                    shoulder_mid = midpoint(kpts[L_SHOULDER], kpts[R_SHOULDER])
                    hip_mid = midpoint(kpts[L_HIP], kpts[R_HIP])

                    torso_angle = calc_torso_angle(shoulder_mid, hip_mid)
                    tracker.update_hip(hip_mid)
                    v_velocity = tracker.vertical_velocity()

                    raw_posture_flag = (
                        aspect_ratio > ASPECT_RATIO_THRESHOLD
                        or torso_angle > TORSO_ANGLE_THRESHOLD
                        or v_velocity > VERTICAL_VELOCITY_THRESHOLD
                    )
                else:
                    # Fallback jika keypoint tertutup saat tergeletak di lantai
                    if aspect_ratio > 0.75:
                        raw_posture_flag = True

                posture_indicates_fall = tracker.update_posture_flag(raw_posture_flag)

                status_text = "Normal"
                if posture_indicates_fall:
                    status_text = "Suspected Fall"
                    if tracker.fall_suspected_since is None:
                        tracker.fall_suspected_since = time.time()

                    duration = time.time() - tracker.fall_suspected_since
                    inactive = tracker.is_inactive(hip_mid)

                    if duration >= INACTIVITY_SECONDS and inactive and not tracker.alert_sent:
                        status_text = "FALL DETECTED"
                        send_alert(frame, track_id, confidence=0.9)
                        tracker.alert_sent = True
                    elif tracker.alert_sent:
                        status_text = "FALL DETECTED"
                else:
                    tracker.fall_suspected_since = None
                    tracker.alert_sent = False

                x1, y1 = int(box[0]), int(box[1])
                color = (0, 0, 255) if "FALL" in status_text else (0, 255, 0)
                cv2.putText(
                    annotated_frame, f"ID {track_id}: {status_text}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                )

        cv2.imshow("Fall Detection - press 'q' to quit", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()