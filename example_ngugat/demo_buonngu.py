import cv2
import dlib
import time
import numpy as np
import threading
from ultralytics import YOLO

class VideoStream:
    """Class đọc luồng video trên một thread riêng biệt để tránh blocking IO."""
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if not grabbed:
                self.stopped = True
                break
            self.frame = frame

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()


class DriverMonitoringSystem:
    def __init__(self, yolo_path, dlib_predictor_path, camera_source):
        self.camera_source = camera_source
        
        print("[INFO] Loading YOLO model...")
        self.yolo_model = YOLO(yolo_path)
        
        print("[INFO] Loading Dlib Face Detector & Shape Predictor...")
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(dlib_predictor_path)
        
        self.EAR_THRESHOLD = 0.25
        self.SLEEP_TIME_THRESH = 2.0
        self.DLIB_SCALE = 0.5  # Scale frame xuống 50% để Dlib chạy mượt hơn
        
        self.LEFT_EYE_IDXS = (42, 48)
        self.RIGHT_EYE_IDXS = (36, 42)
        
        self.eye_closed_start_time = None
        
    def _calculate_ear(self, eye_pts):
        A = np.linalg.norm(eye_pts[1] - eye_pts[5])
        B = np.linalg.norm(eye_pts[2] - eye_pts[4])
        C = np.linalg.norm(eye_pts[0] - eye_pts[3])
        return (A + B) / (2.0 * C)

    def _get_eye_landmarks(self, gray_frame, face):
        shape = self.shape_predictor(gray_frame, face)
        coords = np.zeros((68, 2), dtype="int")
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
            
        left_eye = coords[self.LEFT_EYE_IDXS[0]:self.LEFT_EYE_IDXS[1]]
        right_eye = coords[self.RIGHT_EYE_IDXS[0]:self.RIGHT_EYE_IDXS[1]]
        
        ear_left = self._calculate_ear(left_eye)
        ear_right = self._calculate_ear(right_eye)
        
        return (ear_left + ear_right) / 2.0, left_eye, right_eye

    def run(self):
        print(f"[INFO] Connecting to Camera Stream: {self.camera_source}...")
        vs = VideoStream(self.camera_source).start()
        time.sleep(1.0) # Đợi camera buffer frame
        
        if vs.frame is None:
            print("[ERROR] Cannot connect to camera stream.")
            vs.stop()
            return

        cv2.namedWindow("Cabin Monitoring System", cv2.WINDOW_NORMAL)
        print("[INFO] System is running. Press 'q' to quit.")

        prev_time = time.time()

        while True:
            frame = vs.read()
            if frame is None:
                break

            warnings = []
            
            # Tính toán FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            # --- MODULE 1: YOLO DETECTIONS ---
            results = self.yolo_model.predict(source=frame, conf=0.5, verbose=False)
            annotated_frame = results[0].plot()
            detected_classes = [self.yolo_model.names[int(box.cls[0])] for box in results[0].boxes]
            
            if "phone" in detected_classes:
                warnings.append("WARNING: PHONE USAGE")
            if "yawn" in detected_classes:
                warnings.append("WARNING: YAWNING")

            # --- MODULE 2: DLIB EAR CALCULATION ---
            # Resize frame để tăng tốc độ Dlib
            small_frame = cv2.resize(frame, (0, 0), fx=self.DLIB_SCALE, fy=self.DLIB_SCALE)
            gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector(gray_small, 0)
            
            for face in faces:
                # Map tọa độ face bbox về frame gốc
                l, t, r, b = (int(face.left() / self.DLIB_SCALE), 
                              int(face.top() / self.DLIB_SCALE), 
                              int(face.right() / self.DLIB_SCALE), 
                              int(face.bottom() / self.DLIB_SCALE))
                
                original_face = dlib.rectangle(l, t, r, b)
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                ear, left_eye, right_eye = self._get_eye_landmarks(gray_frame, original_face)
                
                cv2.polylines(annotated_frame, [left_eye], True, (0, 255, 0), 1)
                cv2.polylines(annotated_frame, [right_eye], True, (0, 255, 0), 1)
                cv2.putText(annotated_frame, f"EAR: {ear:.2f}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                if ear < self.EAR_THRESHOLD:
                    if self.eye_closed_start_time is None:
                        self.eye_closed_start_time = time.time()
                    else:
                        elapsed_time = time.time() - self.eye_closed_start_time
                        if elapsed_time >= self.SLEEP_TIME_THRESH:
                            warnings.append("WARNING: SLEEPING!")
                else:
                    self.eye_closed_start_time = None

            # --- HIỂN THỊ KẾT QUẢ ---
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if warnings:
                y_offset = 90
                for warning in set(warnings):
                    cv2.putText(annotated_frame, warning, (10, y_offset), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                    y_offset += 30

            cv2.imshow("Cabin Monitoring System", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        vs.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    YOLO_WEIGHTS = "best.pt"
    DLIB_PREDICTOR = "shape_predictor_68_face_landmarks.dat"
    APP_CAMERA_URL = "http://192.168.2.15:8080/video" 
    
    system = DriverMonitoringSystem(
        yolo_path=YOLO_WEIGHTS,
        dlib_predictor_path=DLIB_PREDICTOR,
        camera_source=APP_CAMERA_URL
    )
    system.run()