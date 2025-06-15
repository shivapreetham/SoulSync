import gradio as gr
import cv2
import mediapipe as mp
import numpy as np
from vosk import Model, KaldiRecognizer
import soundfile as sf
import requests
from collections import Counter
import json
import time
import threading
import queue
import os
import pyaudio
from PIL import Image

# Initialize MediaPipe Face Mesh with optimized settings
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

# Initialize Vosk model
VOSK_MODEL_PATH = "vosk-model-en-us-0.42-gigaspeech"
if not os.path.isdir(VOSK_MODEL_PATH):
    raise RuntimeError(f"Vosk model not found at {VOSK_MODEL_PATH}. Download from https://alphacephei.com/vosk/models")
vosk_model = Model(VOSK_MODEL_PATH)

# Global variables
recording = False
snapshots = []
audio_path = None
audio_thread = None
audio_queue = queue.Queue()
cap = None
last_frame = None
last_snapshot_time = 0

def find_available_camera(max_index=5):
    """Find the first available camera by trying different indices."""
    for index in range(max_index):
        temp_cap = cv2.VideoCapture(index)
        if temp_cap.isOpened():
            ret, frame = temp_cap.read()
            if ret and frame is not None and frame.size > 0:
                print(f"Debug: Camera found at index {index}")
                return temp_cap
            temp_cap.release()
    print("Debug: No camera found")
    return None

def get_landmarks(frame):
    """Extract facial landmarks from a frame."""
    if frame is None:
        print("Debug: get_landmarks received None frame")
        return None
        
    # Flip frame for natural selfie view
    frame = cv2.flip(frame, 1)
    
    # Convert to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    try:
        results = face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            print("Debug: No face landmarks detected")
            return None
        
        lm = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape
        coords = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
        print(f"Debug: Landmarks extracted - {len(coords)} points")
        return coords
    except Exception as e:
        print(f"Error in get_landmarks: {str(e)}")
        return None

def mouth_aspect_ratio(lm):
    # Updated landmark indices for better accuracy
    left = lm[78]    # Left mouth corner
    right = lm[308]  # Right mouth corner
    top = lm[13]     # Top lip center
    bottom = lm[14]  # Bottom lip center
    
    width = np.linalg.norm(right - left)
    height = np.linalg.norm(top - bottom)
    return height / width if width > 0 else 0.0

def eye_aspect_ratio(lm, left=True):
    if left:
        # Left eye landmarks
        upper = lm[159]  # Upper eyelid
        lower = lm[145]  # Lower eyelid
        corner1 = lm[33] # Outer corner
        corner2 = lm[133] # Inner corner
    else:
        # Right eye landmarks
        upper = lm[386]  # Upper eyelid
        lower = lm[374]  # Lower eyelid
        corner1 = lm[263] # Outer corner
        corner2 = lm[362] # Inner corner
        
    vert = np.linalg.norm(upper - lower)
    hor = np.linalg.norm(corner2 - corner1)
    return vert / hor if hor > 0 else 0.0

def eyebrow_distance(lm):
    # Updated landmark indices
    left_eyebrow = lm[70]    # Left eyebrow inner
    right_eyebrow = lm[300]  # Right eyebrow inner
    return np.linalg.norm(left_eyebrow - right_eyebrow)

def mouth_corner_slope(lm):
    # Updated landmark indices
    left = lm[78]    # Left mouth corner
    right = lm[308]  # Right mouth corner
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    return dy / dx if dx != 0 else 0.0

def detect_emotion_from_frame(frame):
    """Detect emotion from facial landmarks with improved logic."""
    lm = get_landmarks(frame)
    if lm is None:
        return "neutral"
    
    try:
        mar = mouth_aspect_ratio(lm)
        left_ear = eye_aspect_ratio(lm, left=True)
        right_ear = eye_aspect_ratio(lm, left=False)
        ear = (left_ear + right_ear) / 2.0
        brow_dist = eyebrow_distance(lm)
        slope = mouth_corner_slope(lm)
        
        # Use face width for relative measurements
        face_width = np.linalg.norm(lm[234] - lm[454])
        
        # Emotion thresholds (tuned values)
        if mar > 0.15:  # Smiling (mouth open)
            return "happy"
        elif ear > 0.25:  # Wide open eyes
            return "surprised"
        elif brow_dist < 0.15 * face_width:  # Furrowed brows
            return "angry"
        elif slope > 0.2:  # Downward mouth corners
            return "sad"
        return "neutral"
    except Exception as e:
        print(f"Error in emotion detection: {str(e)}")
        return "neutral"

def record_audio():
    """Record audio in a separate thread."""
    global audio_path
    try:
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, 
                        channels=1, 
                        rate=16000, 
                        input=True, 
                        frames_per_buffer=1024)
        frames = []
        print("Recording audio...")
        
        while recording:
            try:
                data = stream.read(1024, exception_on_overflow=False)
                frames.append(data)
            except OSError as e:
                print(f"Audio read error: {e}")
                continue
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        if frames:
            audio_data = b''.join(frames)
            audio_path = "temp_audio.wav"
            try:
                sf.write(audio_path, np.frombuffer(audio_data, dtype=np.int16), 16000)
                print(f"Audio saved at: {audio_path}")
                audio_queue.put(audio_path)
            except Exception as e:
                print(f"Audio save error: {e}")
                audio_queue.put(None)
        else:
            print("No audio frames recorded")
            audio_queue.put(None)
    except Exception as e:
        print(f"Error in audio recording: {str(e)}")
        audio_queue.put(None)

def video_stream():
    """Stream video frames as PIL images and capture snapshots."""
    global cap, last_frame, last_snapshot_time
    cap = find_available_camera()
    if cap is None:
        print("Error: No available webcam found")
        while True:
            yield Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
            time.sleep(0.033)

    # Set camera resolution to 640x480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Print actual camera resolution
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Debug: Camera resolution set to: {actual_width}x{actual_height}")

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                if last_frame is not None:
                    yield Image.fromarray(cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB))
                else:
                    yield Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
                time.sleep(0.033)
                continue
                
            # Flip for selfie view
            frame = cv2.flip(frame, 1)
            
            # Resize to target resolution
            frame = cv2.resize(frame, (640, 480))
            
            # Process frame for landmarks visualization
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            # Draw landmarks if detected
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            
            # Save for next frame if this one fails
            last_frame = frame.copy()
            
            # Convert to RGB for PIL
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            yield pil_image

            # Capture snapshots for emotion detection
            current_time = time.time()
            if recording and current_time - last_snapshot_time >= 1.0:
                # Save original frame (without landmarks) for emotion detection
                snapshots.append(frame.copy())
                last_snapshot_time = current_time
                print(f"Debug: Snapshot captured at {current_time}")
                
        except Exception as e:
            print(f"Video stream error: {str(e)}")
            yield Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
        time.sleep(0.033)

def start_recording():
    global recording, snapshots, audio_thread, last_snapshot_time
    if recording:
        return "Already recording"
    
    recording = True
    snapshots = []
    last_snapshot_time = time.time()
    audio_thread = threading.Thread(target=record_audio, daemon=True)
    audio_thread.start()
    return "Recording audio and video... Speak clearly for 10–15 seconds."

def stop_and_process():
    global recording, audio_path, audio_thread
    if not recording:
        return "No active recording", "No transcription", "No server response"
    
    recording = False
    if audio_thread:
        audio_thread.join(timeout=2.0)
        audio_thread = None
    
    try:
        audio_path = audio_queue.get(timeout=3.0)
    except queue.Empty:
        audio_path = None
        print("No audio recorded")

    # Process emotions
    emotions = []
    for i, frame in enumerate(snapshots):
        try:
            emotion = detect_emotion_from_frame(frame)
            if emotion:
                emotions.append(emotion)
                print(f"Snapshot {i}: Emotion detected: {emotion}")
            else:
                print(f"Snapshot {i}: No emotion detected")
        except Exception as e:
            print(f"Error processing emotion in snapshot {i}: {e}")
            emotions.append("neutral")
    
    most_common_emotion = Counter(emotions).most_common(1)[0][0] if emotions else "neutral"
    print(f"Most common emotion: {most_common_emotion}")

    # Transcribe audio
    text = "No audio recorded"
    if audio_path and os.path.exists(audio_path):
        try:
            data, samplerate = sf.read(audio_path)
            if data.ndim > 1:
                data = data[:, 0]  # Use first channel if stereo
                
            # Ensure data is in 16-bit PCM format
            if data.dtype != np.int16:
                data = (data * 32767).astype(np.int16)
                
            recognizer = KaldiRecognizer(vosk_model, samplerate)
            recognizer.AcceptWaveform(data.tobytes())
            result = json.loads(recognizer.FinalResult())
            text = result.get("text", "No speech detected")
            print(f"Transcription: {text}")
        except Exception as e:
            text = f"Transcription error: {str(e)}"
            print(f"Transcription error: {str(e)}")
    else:
        print("Audio file missing")

    # Send to Gemini server
    response_text = "No server response"
    if text and "error" not in text.lower():
        payload = {"message": text}
        try:
            response = requests.post("https://sign-language-3-5vax.onrender.com/gemini", 
                                    json=payload, 
                                    timeout=10)
            response.raise_for_status()
            response_text = response.json().get("response", response.text)
            print(f"Server response: {response_text}")
        except Exception as e:
            response_text = f"Server error: {str(e)}"
            print(f"Server error: {response_text}")

    # Cleanup
    if audio_path and os.path.exists(audio_path):
        try:
            os.remove(audio_path)
        except:
            pass

    return most_common_emotion, text, response_text

def get_recording_status():
    return "Recording audio and video..." if recording else "Not recording"

def cleanup():
    global cap
    if cap and cap.isOpened():
        cap.release()
    cap = None

with gr.Blocks(title="Video Call App", delete_cache=(60, 3600)) as demo:
    gr.Markdown("## Video Call App")
    gr.Markdown("Click 'Start Recording' to begin audio and video. Record for 10–15 seconds, speaking clearly, then click 'Stop and Process'.")
    image_out = gr.Image(label="Live Webcam", streaming=True)
    status_out = gr.Textbox(label="Recording Status", value="Not recording", interactive=False)
    start_btn = gr.Button("Start Recording")
    stop_btn = gr.Button("Stop and Process")
    emotion_out = gr.Textbox(label="Detected Emotion", interactive=False)
    transcript_out = gr.Textbox(label="Transcription", interactive=False)
    response_out = gr.Textbox(label="Server Response", interactive=False)
    timer = gr.Timer(value=1.0, active=True)
    start_btn.click(fn=start_recording, outputs=status_out)
    stop_btn.click(fn=stop_and_process, outputs=[emotion_out, transcript_out, response_out])
    timer.tick(fn=get_recording_status, outputs=status_out)
    demo.load(video_stream, None, image_out)
    demo.unload(cleanup)

if __name__ == "__main__":
    demo.launch()
