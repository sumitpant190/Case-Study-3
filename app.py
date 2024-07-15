import streamlit as st
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
from datetime import datetime
import pygame

class ObjectDetection:
    CUSTOM_CLASSES = ['bicycle', 'bus', 'car', 'motorbike', 'person']
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using Device: ", self.device)
        self.model = self.load_model()
        
    def load_model(self):
        model = YOLO("best.pt")
        model.fuse()
        return model
    
    def predict(self, image):
        results = self.model(image)
        return results
    
    def plot_bboxes(self, results, image):
        xyxys = []
        confidences = []
        class_ids = []

        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                xyxy = box.xyxy[0]
                conf = box.conf[0]
                cls = int(box.cls[0])
                
                xyxys.append(xyxy)
                confidences.append(conf)
                class_ids.append(cls)
                
                cv2.rectangle(image, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                if int(cls) < len(self.CUSTOM_CLASSES):
                    label = f"{self.CUSTOM_CLASSES[int(cls)]}: {conf:.2f}"
                else:
                    label = f"Unknown: {conf:.2f}"
                cv2.putText(image, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image_rgb, xyxys, confidences, class_ids, results

    def get_inference_info(self, results):
        class_counts = {class_name: 0 for class_name in self.CUSTOM_CLASSES}
        total_vehicles = 0

        for result in results:
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            for cls_id in class_ids:
                class_name = self.CUSTOM_CLASSES[cls_id]
                class_counts[class_name] += 1
                if class_name in ['bicycle', 'bus', 'car', 'motorbike']:
                    total_vehicles += 1

        info = ""
        for class_name, count in class_counts.items():
            info += f"{class_name}: {count}\n"
        info += f"Total Vehicles: {total_vehicles}\n"
        info += f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

        return info, total_vehicles

def play_audio_alert():
    try:
        pygame.mixer.init()
        pygame.mixer.music.load("alert_sound.mp3")  # Replace with your audio file path
        pygame.mixer.music.play()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        st.success(f"Audio Alert Played at {timestamp}")  # Display audio alert with timestamp in main area
    except Exception as e:
        st.error(f"Error playing audio: {e}")

def main():
    st.title("Smart City Traffic Management")
    
    uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])
    if uploaded_file is not None:
        if uploaded_file.type.startswith("image"):
            image = Image.open(uploaded_file)
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            detector = ObjectDetection()
            
            if st.button("Predict"):
                results = detector.predict(image)
                image_rgb, _, _, _, results = detector.plot_bboxes(results, image)

                # Display image in main area
                st.image(image_rgb, channels="RGB", caption="Predicted Image")

                # Display inference info below the image
                inference_placeholder = st.empty()  # Placeholder for dynamic updating
                for i, result in enumerate(results):
                    if i % 3 == 0:  # Display for every 3rd frame
                        inference_info, total_vehicles = detector.get_inference_info([result])
                        inference_placeholder.text(inference_info)
                        if total_vehicles >= 8:
                            play_audio_alert()

        elif uploaded_file.type.startswith("video"):
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_file.read())

            cap = cv2.VideoCapture(tfile.name)

            detector = ObjectDetection()
            frame_count = 0

            stframe = st.empty()
            inference_placeholder = st.empty()
            sidebar_alerts = st.sidebar.empty()
            sidebar_inference = st.sidebar.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                if frame_count % 3 == 0:  # Process every 3rd frame
                    results = detector.predict(frame)
                    frame_rgb, _, _, _, results = detector.plot_bboxes(results, frame)
                    
                    stframe.image(frame_rgb, channels="RGB", caption=f"Processed Frame {frame_count}")

                    # Update inference info below the video for every 3rd frame
                    inference_info, total_vehicles = detector.get_inference_info(results)
                    inference_placeholder.text(inference_info)

                    if total_vehicles >= 7:
                        # Display alert and corresponding inference info in sidebar
                        sidebar_alerts.text("Congestion Detected!")
                        sidebar_inference.text(inference_info)
                        play_audio_alert()

            cap.release()

if __name__ == "__main__":
    main()
