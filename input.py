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

# Initialize MediaPipe Face Mesh with lower detection confidence
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.2,  # Lowered for better sensitivity
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
last_frame = None
last_snapshot_time = 0

def find_available_camera(max_index=5):
    """Find the first available camera by trying different indices."""
    for index in range(max_index):
        temp_cap = cv2.VideoCapture(index)
        if temp_cap.isOpened():
            print(f"Debug: Camera found at index {index}")
            return temp_cap
        temp_cap.release()
    return None

def get_landmarks(frame):
    """Extract facial landmarks from a frame."""
    rgb = frame  # Frame is already in RGB
    try:
        results = face_mesh.process(rgb)
        print(f"Debug: Frame shape {rgb.shape}, type {rgb.dtype}")
        if not results.multi_face_landmarks:
            print("Debug: No face landmarks detected")
            return None, frame
        lm = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape
        coords = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
        print(f"Debug: Landmarks detected - {len(coords)} coordinates")
        # Draw landmarks on frame for debugging
        output_frame = frame.copy()
        mp_drawing.draw_landmarks(output_frame, results.multi_face_landmarks[0])
        return coords, output_frame
    except Exception as e:
        print(f"Error in get_landmarks: {e}")
        return None, frame

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
    lm, output_frame = get_landmarks(frame)
    if lm is None:
        return None, output_frame
    if mouth_aspect_ratio(lm) > thresh_smile:
        return "happy", output_frame
    ear = (eye_aspect_ratio(lm, True) + eye_aspect_ratio(lm, False)) / 2.0
    if ear > thresh_surprise:
        return "surprised", output_frame
    face_width = np.linalg.norm(lm[234] - lm[454])
    if face_width > 0 and eyebrow_distance(lm) < thresh_anger * face_width:
        return "angry", output_frame
    if mouth_corner_slope(lm) > thresh_sad_slope:
        return "sad", output_frame
    return "neutral", output_frame

def record_audio():
    global audio_path
    try:
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
    except Exception as e:
        print(f"Audio recording error: {e}")

def video_stream():
    """Stream video frames as PIL images and capture snapshots."""
    global cap, last_frame, last_snapshot_time
    cap = find_available_camera()
    if cap is None:
        print("Error: No available webcam found")
        while True:
            yield Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
            time.sleep(0.033)

    print(f"Camera resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        try:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (640, 480))
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                elif frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                last_frame = frame.copy()
                pil_image = Image.fromarray(frame)
                yield pil_image

                current_time = time.time()
                if recording and current_time - last_snapshot_time >= 0.5:  # Increased frequency
                    snapshots.append(frame.copy())
                    print(f"Debug: Snapshot {len(snapshots)} captured at {current_time}")
                    last_snapshot_time = current_time
            else:
                if last_frame is not None:
                    yield Image.fromarray(last_frame)
                else:
                    yield Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
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
    emotions = []
    for i, frame in enumerate(snapshots):
        try:
            emotion, output_frame = detect_emotion_from_frame(frame)
            cv2.imwrite(f"snapshot_{i}_processed.jpg", cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))
            print(f"Debug: Saved processed snapshot_{i}_processed.jpg")
            if emotion:
                emotions.append(emotion)
                print(f"Debug: Emotion '{emotion}' detected in snapshot {i}")
            else:
                print(f"Debug: No face detected in snapshot {i}")
        except Exception as e:
            print(f"Error processing snapshot {i}: {e}")
    most_common_emotion = Counter(emotions).most_common(1)[0][0] if emotions else "No face detected"
    print(f"Detected emotions: {emotions}, Most common: {most_common_emotion}")

    # Transcribe audio
    print("Transcribing audio...")
    try:
        data, samplerate = sf.read(audio_path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = (data * 32767).astype(np.int16)
        recognizer = KaldiRecognizer(vosk_model, samplerate)
        if not recognizer.AcceptWaveform(data.tobytes()):
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
    demo.load(fn=video_stream, outputs=image_out)
    demo.unload(cleanup)

if __name__ == "__main__":
    demo.launch()