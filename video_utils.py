import cv2
import mediapipe as mp
from PIL import Image
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def find_available_camera(max_index=5):
    for index in range(max_index):
        temp_cap = cv2.VideoCapture(index)
        if temp_cap.isOpened():
            ret, frame = temp_cap.read()
            if ret and frame is not None and frame.size > 0:
                return temp_cap
            temp_cap.release()
    return None

def video_stream():
    global cap, last_frame, last_snapshot_time, recording, snapshots
    cap = find_available_camera()
    if cap is None:
        while True:
            yield Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
            time.sleep(0.033)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            if last_frame is not None:
                yield Image.fromarray(cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB))
            else:
                yield Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
            time.sleep(0.033)
            continue
        
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
        
        last_frame = frame.copy()
        yield Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if recording and time.time() - last_snapshot_time >= 1.0:
            snapshots.append(frame.copy())
            last_snapshot_time = time.time()
        
        time.sleep(0.033)