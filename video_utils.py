# #updated
# Moved video-related functions from main.py.
# Separated frame processing and streaming logic.
# Handles camera initialization and cleanup.
# Removed global variables (cap, last_frame) to make the module stateless.
# Imported snapshots, recording, and last_snapshot_time from main.py to access global state.
# Added snapshot capture every second when recording is True.
# Added error handling in process_video_frame call.
# Added debug print to confirm snapshot capture (replace with logger.info later).
import cv2
import numpy as np
from PIL import Image
import time
import mediapipe as mp
from typing import Generator, Optional
from logging_utils import setup_logger

logger = setup_logger("video_utils")

# Initialize MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

def find_available_camera(max_index: int = 5, backends=None) -> Optional[cv2.VideoCapture]:
    """Find an available camera device with specified backends."""
    if backends is None:
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]  # Try multiple backends
    for backend in backends:
        for index in range(max_index):
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                # Warm-up: read a few frames to stabilize
                for _ in range(10):
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        logger.info(f"Camera found at index {index} with backend {backend}")
                        return cap
                    time.sleep(0.01)
                cap.release()
            time.sleep(0.1)
    logger.warning(f"No camera found after checking indices 0 to {max_index - 1} with backends {backends}")
    return None

def process_video_frame(cap: cv2.VideoCapture, face_mesh, retry_count: int = 0) -> Optional[np.ndarray]:
    """Process a single video frame with face mesh overlay."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning(f"Failed to read frame from camera (attempt {attempt + 1}/{max_retries})")
                time.sleep(0.05)
                continue
            
            logger.debug(f"Successfully read frame (attempt {attempt + 1})")
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (640, 480))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = True
            results = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1)
                    )
            
            return frame
        except Exception as e:
            logger.error(f"Frame processing error (attempt {attempt + 1}/{max_retries}): {str(e)}")
            time.sleep(0.05)
    
    # Reinitialize camera if all retries fail
    if retry_count < 1:
        logger.warning("Reinitializing camera after failed retries")
        cap.release()
        cap = find_available_camera()
        if cap is not None:
            return process_video_frame(cap, face_mesh, retry_count + 1)
    
    logger.error("Failed to process frame after {} retries".format(max_retries))
    return None

def video_stream(snapshots: list, recording: bool, last_snapshot_time: float) -> Generator[Image.Image, None, None]:
    """Generate a video stream for Gradio interface."""
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    cap = find_available_camera()
    last_frame = None
    local_last_snapshot_time = last_snapshot_time
    
    if cap is None:
        logger.warning("No camera found for video stream")
        while True:
            yield Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
            time.sleep(0.033)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    try:
        while True:
            frame = process_video_frame(cap, face_mesh)
            if frame is None and last_frame is not None:
                yield Image.fromarray(cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB))
            elif frame is None:
                yield Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
            elif frame is not None:
                last_frame = frame.copy()
                yield Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if recording and time.time() - local_last_snapshot_time >= 1.0:
                    snapshots.append(frame.copy())
                    local_last_snapshot_time = time.time()
                    logger.info(f"Snapshot captured. Total snapshots: {len(snapshots)}")
            time.sleep(0.033)
    finally:
        if cap is not None:
            cap.release()
        face_mesh.close()