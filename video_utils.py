import cv2
import numpy as np
import mediapipe as mp
from collections import Counter
from PIL import Image

class VideoProcessor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.cap = None
        self.last_frame = None

    def find_camera(self, max_index=5):
        for index in range(max_index):
            temp_cap = cv2.VideoCapture(index)
            if temp_cap.isOpened():
                ret, frame = temp_cap.read()
                if ret and frame is not None:
                    self.cap = temp_cap
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    return True
                temp_cap.release()
        return False

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return cv2.flip(frame, 1)

    def process_frame(self, frame):
        if frame is None:
            return Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
        
        # Convert to RGB for processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        # Convert back to PIL Image for display
        self.last_frame = frame.copy()
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def detect_emotion(self, frame):
        # Get facial landmarks
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            return "neutral"
            
        # Extract landmarks
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]
        lm = np.array([[p.x * w, p.y * h] for p in landmarks])
        
        # Calculate emotion features
        mar = self._mouth_aspect_ratio(lm)
        ear = (self._eye_aspect_ratio(lm, True) + self._eye_aspect_ratio(lm, False)) / 2
        brow_dist = self._eyebrow_distance(lm)
        slope = self._mouth_corner_slope(lm)
        
        # Determine emotion
        if mar > 0.15: return "happy"
        elif ear > 0.25: return "surprised"
        elif brow_dist < 0.15 * np.linalg.norm(lm[234] - lm[454]): return "angry"
        elif slope > 0.2: return "sad"
        return "neutral"

    def _mouth_aspect_ratio(self, lm):
        left, right, top, bottom = lm[78], lm[308], lm[13], lm[14]
        width = np.linalg.norm(right - left)
        height = np.linalg.norm(top - bottom)
        return height / width if width > 0 else 0.0

    def _eye_aspect_ratio(self, lm, left=True):
        indices = [159, 145, 33, 133] if left else [386, 374, 263, 362]
        vert = np.linalg.norm(lm[indices[0]] - lm[indices[1]])
        hor = np.linalg.norm(lm[indices[3]] - lm[indices[2]])
        return vert / hor if hor > 0 else 0.0

    def _eyebrow_distance(self, lm):
        return np.linalg.norm(lm[70] - lm[300])

    def _mouth_corner_slope(self, lm):
        left, right = lm[78], lm[308]
        dx, dy = right[0] - left[0], right[1] - left[1]
        return dy / dx if dx != 0 else 0.0

    def cleanup(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.face_mesh.close()