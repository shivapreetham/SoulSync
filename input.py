#this is working
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

# Initialize MediaPipe Face Mesh
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize Vosk model
VOSK_MODEL_PATH = "vosk-model-en-us-0.22"
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

def get_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None
    lm = results.multi_face_landmarks[0].landmark
    h, w, _ = frame.shape
    coords = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
    return coords

def mouth_aspect_ratio(lm):
    left, right = lm[61], lm[291]
    top, bottom = lm[13], lm[14]
    width = np.linalg.norm(right - left)
    height = np.linalg.norm(top - bottom)
    return 0.0 if width == 0 else height / width

def eye_aspect_ratio(lm, left=True):
    if left:
        upper, lower = lm[159], lm[145]
        corner1, corner2 = lm[33], lm[133]
    else:
        upper, lower = lm[386], lm[374]
        corner1, corner2 = lm[263], lm[362]
    vert = np.linalg.norm(upper - lower)
    hor = np.linalg.norm(corner2 - corner1)
    return 0.0 if hor == 0 else vert / hor

def eyebrow_distance(lm):
    return np.linalg.norm(lm[70] - lm[300])

def mouth_corner_slope(lm):
    left, right = lm[61], lm[291]
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    return 0.0 if dx == 0 else dy / dx

def detect_emotion_from_frame(frame, thresh_smile=0.035, thresh_surprise=0.25, thresh_anger=0.05, thresh_sad_slope=0.02):
    lm = get_landmarks(frame)
    if lm is None:
        return None
    if mouth_aspect_ratio(lm) > thresh_smile:
        return "happy"
    ear = (eye_aspect_ratio(lm, True) + eye_aspect_ratio(lm, False)) / 2.0
    if ear > thresh_surprise:
        return "surprised"
    face_width = np.linalg.norm(lm[234] - lm[454])
    if face_width > 0 and eyebrow_distance(lm) < thresh_anger * face_width:
        return "angry"
    if mouth_corner_slope(lm) > thresh_sad_slope:
        return "sad"
    return "neutral"

def record_audio():
    global audio_path
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    frames = []
    print("Recording audio...")
    while recording:
        data = stream.read(1024, exception_on_overflow=False)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()
    audio_data = b''.join(frames)
    audio_path = "temp_audio.wav"
    sf.write(audio_path, np.frombuffer(audio_data, dtype=np.int16), 16000)
    audio_queue.put(audio_path)
    print(f"Audio saved at: {audio_path}")

def capture_snapshots():
    global snapshots, cap
    if not cap.isOpened():
        print("Error: Webcam not accessible")
        return
    last_time = time.time()
    print("Capturing snapshots...")
    while recording:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        current_time = time.time()
        if current_time - last_time >= 1.0:
            snapshots.append(frame.copy())
            last_time = current_time

def video_stream():
    global cap
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    # Optimize: Lower resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame
        time.sleep(0.05)  # ~20 FPS , use 0.01 for faster frames

def start_recording():
    global recording, snapshots, audio_thread
    if recording:
        return "Already recording"
    recording = True
    snapshots = []
    threading.Thread(target=capture_snapshots, daemon=True).start()
    audio_thread = threading.Thread(target=record_audio, daemon=True)
    audio_thread.start()
    return "Recording audio and video... Speak clearly for 10–15 seconds."

def stop_and_process():
    global recording, audio_path, audio_thread
    print("Stopping recording...")
    if not recording:
        print("No active recording")
        return "No active recording", "No transcription", "No server response"
    recording = False
    if audio_thread:
        audio_thread.join()
        audio_thread = None
    try:
        audio_path = audio_queue.get_nowait()
        print(f"Audio saved at: {audio_path}")
    except queue.Empty:
        print("No audio recorded")
        return "No face detected", "No audio recorded", "No server response"

    # Process emotions
    print("Processing emotions...")
    emotions = [detect_emotion_from_frame(frame) for frame in snapshots]
    emotions = [e for e in emotions if e is not None]
    most_common_emotion = Counter(emotions).most_common(1)[0][0] if emotions else "No face detected"
    print(f"Detected emotions: {emotions}, Most common: {most_common_emotion}")

    # Transcribe audio
    print("Transcribing audio...")
    try:
        data, samplerate = sf.read(audio_path)
        print(f"Audio shape: {data.shape}, Sample rate: {samplerate}")
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = (data * 32767).astype(np.int16)
        recognizer = KaldiRecognizer(vosk_model, samplerate)
        if not recognizer.AcceptWaveform(data.tobytes()):
            print("Vosk failed to process audio")
            text = "No speech detected"
        else:
            result = json.loads(recognizer.Result())
            text = result.get("text", "") or "No speech detected"
        print(f"Transcription: {text}")
    except Exception as e:
        print(f"Transcription error: {str(e)}")
        text = f"Transcription error: {str(e)}"

    # Send to Gemini server
    print("Sending to Gemini server...")
    payload = {"message": text}
    try:
        response = requests.post("https://sign-language-3-5vax.onrender.com/gemini", json=payload, timeout=5)
        response.raise_for_status()
        response_text = response.json().get("response", response.text)
        print(f"Server response: {response_text}")
    except Exception as e:
        response_text = f"Server error: {str(e)}"
        print(f"Server error: {response_text}")

    return most_common_emotion, text, response_text

def get_recording_status():
    return "Recording audio and video..." if recording else "Not recording"

def cleanup():
    global cap
    if cap and cap.isOpened():
        cap.release()

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

demo.launch()