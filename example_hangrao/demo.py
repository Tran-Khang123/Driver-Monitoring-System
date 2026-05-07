import cv2
import os
from ultralytics import YOLO

MODEL_PATH = "best.pt"
VIDEO_PATH = "VID_20260417165334187.mp4"
OUTPUT_FOLDER = "/home/rtx5070/Desktop/dataset_vat_tu_cong_truong/roi_detections"
CONF_THRESHOLD = 0.35

class ROIDetector:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        self.roi_x1, self.roi_y1 = 0, 0
        self.roi_x2, self.roi_y2 = 0, 0
        self.drawing = False
        self.roi_set = False
        self.video_path = VIDEO_PATH
        self.class_names = self.model.names
        self.play_clicked = False
        
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.roi_set:
                play_x1, play_x2 = 50, 150
                play_y1, play_y2 = 20, 60
                if play_x1 <= x <= play_x2 and play_y1 <= y <= play_y2:
                    self.play_clicked = True
                    return
            self.drawing = True
            self.roi_x1, self.roi_y1 = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.roi_x2, self.roi_y2 = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                self.roi_x2, self.roi_y2 = x, y
                self.roi_set = True
            
    def draw_roi(self, frame):
        if self.drawing and not self.roi_set:
            cv2.rectangle(frame, (self.roi_x1, self.roi_y1), (self.roi_x2, self.roi_y2), (255, 0, 0), 2)
        elif self.roi_set:
            x1, y1 = min(self.roi_x1, self.roi_x2), min(self.roi_y1, self.roi_y2)
            x2, y2 = max(self.roi_x1, self.roi_x2), max(self.roi_y1, self.roi_y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, "ROI", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        return frame
        
    def is_in_roi(self, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        roi_x1, roi_y1 = min(self.roi_x1, self.roi_x2), min(self.roi_y1, self.roi_y2)
        roi_x2, roi_y2 = max(self.roi_x1, self.roi_x2), max(self.roi_y1, self.roi_y2)
        
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        return roi_x1 <= center_x <= roi_x2 and roi_y1 <= center_y <= roi_y2
        
    def select_roi(self):
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("Error: Cannot read video")
            return False
            
        cv2.namedWindow("Draw ROI box, then click PLAY button")
        cv2.setMouseCallback("Draw ROI box, then click PLAY button", self.mouse_callback)
        
        while True:
            display_frame = frame.copy()
            display_frame = self.draw_roi(display_frame)
            
            if self.roi_set:
                cv2.rectangle(display_frame, (50, 20), (150, 60), (0, 255, 0), -1)
                cv2.rectangle(display_frame, (50, 20), (150, 60), (0, 0, 0), 2)
                cv2.putText(display_frame, "PLAY", (65, 47), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                cv2.putText(display_frame, "Click PLAY to start", (170, 45), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.putText(display_frame, "Draw ROI: Click and drag mouse", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Draw ROI box, then click PLAY button", display_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            
            if self.play_clicked:
                break
                
        cv2.destroyAllWindows()
        return self.roi_set and self.play_clicked
        
    def run_detection(self):
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print("Error: Cannot open video")
            return
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30
        delay = max(1, int(1000 / fps))
        
        frame_count = 0
        saved_count = 0
        
        print(f"Starting detection at {fps:.1f} FPS...")
        print("Press 'q' to stop")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            results = self.model.predict(frame, conf=CONF_THRESHOLD, verbose=False)[0]
            
            alert_triggered = False
            detected_objects = []
            
            if results.boxes is not None:
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = self.class_names[cls_id]
                    
                    if self.is_in_roi([x1, y1, x2, y2]):
                        alert_triggered = True
                        detected_objects.append(f"{cls_name} ({conf:.2f})")
                        color = (0, 0, 255)
                    else:
                        color = (0, 255, 0)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{cls_name} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if self.roi_set:
                x1, y1 = min(self.roi_x1, self.roi_x2), min(self.roi_y1, self.roi_y2)
                x2, y2 = max(self.roi_x1, self.roi_x2), max(self.roi_y1, self.roi_y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            if alert_triggered:
                print(f"Frame {frame_count}: ALERT - Objects in ROI: {', '.join(detected_objects)}")
                save_path = os.path.join(OUTPUT_FOLDER, f"detection_{saved_count:04d}.jpg")
                cv2.imwrite(save_path, frame)
                saved_count += 1
            
            cv2.imshow("Detection", frame)
            
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"Detection complete. Saved {saved_count} frames to {OUTPUT_FOLDER}")
        
    def run(self):
        print("Drawing ROI selection interface...")
        print("Step 1: Draw ROI box by clicking and dragging")
        print("Step 2: Click PLAY button to start detection")
        if self.select_roi():
            print("ROI set. Starting detection...")
            self.run_detection()
        else:
            print("ROI selection cancelled")

if __name__ == "__main__":
    detector = ROIDetector()
    detector.run()