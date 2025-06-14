import gradio as gr
import threading
import queue
import os
import time
import torch
import cv2
import mediapipe as mp
from PIL import Image
import numpy as np
import pyaudio
import soundfile as sf
from vosk import Model, KaldiRecognizer
import json
import pyttsx3
from llm_utils import load_model, generate_response, get_smart_fallback
from chat_utils import build_prompt, truncate_history, validate_response, detect_conversation_quality
from collections import Counter

# Initialize MediaPipe Face Mesh for video and emotion detection
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

# Initialize Vosk model for audio transcription
VOSK_MODEL_PATH = "vosk-model-en-us-0.42-gigaspeech"
if not os.path.isdir(VOSK_MODEL_PATH):
    raise RuntimeError(f"Vosk model not found at {VOSK_MODEL_PATH}")
vosk_model = Model(VOSK_MODEL_PATH)

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed (words per minute)
engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)

# Global variables for video call
recording = False
snapshots = []
audio_thread = None
audio_queue = queue.Queue()
cap = None
last_frame = None
last_snapshot_time = 0

# Global variables for chat state
messages = []
conversation_history = []
tokenizer = None
model = None
device = None
model_name = None

# Model options
model_options = {
    "DialoGPT Large (355M)": "microsoft/DialoGPT-large",
    "BlenderBot 3B": "facebook/blenderbot-3B",
    "BlenderBot 1B": "facebook/blenderbot-1B-distill",
    "GPT-2 Large (774M)": "gpt2-large",
    "GPT-2 XL (1.5B)": "gpt2-xl"
}

# Video processing functions
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
    
    # Create a transparent green color (BGR format with alpha)
    transparent_green = (0, 255, 0, 0.4)  # Green with 40% opacity
    
    # Create drawing specs with transparency
    face_mesh_style = mp_drawing_styles.DrawingSpec(
        color=(0, 255, 0),  # Green color
        thickness=1,
        circle_radius=1
    )
    
    contour_style = mp_drawing_styles.DrawingSpec(
        color=(0, 255, 0),  # Green color
        thickness=1,
        circle_radius=1
    )
    
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
        
        # Create a transparent overlay for the face mesh
        overlay = frame.copy()
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw the face mesh on the overlay with thin, subtle lines
                mp_drawing.draw_landmarks(
                    image=overlay,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=face_mesh_style
                )
                
                # Draw the face contours more subtly
                mp_drawing.draw_landmarks(
                    image=overlay,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=contour_style
                )
        
        # Blend the overlay with the original frame
        alpha = 0.03  # Transparency factor (30% opacity)
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        last_frame = frame.copy()
        yield Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if recording and time.time() - last_snapshot_time >= 1.0:
            snapshots.append(frame.copy())
            last_snapshot_time = time.time()
        
        time.sleep(0.033)

# Audio processing functions
def record_audio():
    global recording, audio_queue
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    frames = []
    
    while recording:
        data = stream.read(1024, exception_on_overflow=False)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    if frames:
        audio_data = b''.join(frames)
        audio_path = "temp_audio.wav"
        sf.write(audio_path, np.frombuffer(audio_data, dtype=np.int16), 16000)
        audio_queue.put(audio_path)
    else:
        audio_queue.put(None)

def transcribe_audio(audio_path):
    if not audio_path or not os.path.exists(audio_path):
        return "No audio recorded"
    try:
        data, samplerate = sf.read(audio_path)
        if data.ndim > 1:
            data = data[:, 0]
        if data.dtype != np.int16:
            data = (data * 32767).astype(np.int16)
        recognizer = KaldiRecognizer(vosk_model, samplerate)
        recognizer.AcceptWaveform(data.tobytes())
        result = json.loads(recognizer.FinalResult())
        return result.get("text", "No speech detected")
    except Exception as e:
        return f"Transcription error: {str(e)}"

# Emotion detection functions
def get_landmarks(frame):
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None
    lm = results.multi_face_landmarks[0].landmark
    h, w, _ = frame.shape
    return np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)

def mouth_aspect_ratio(lm):
    left, right, top, bottom = lm[78], lm[308], lm[13], lm[14]
    width = np.linalg.norm(right - left)
    height = np.linalg.norm(top - bottom)
    return height / width if width > 0 else 0.0

def eye_aspect_ratio(lm, left=True):
    if left:
        upper, lower, corner1, corner2 = lm[159], lm[145], lm[33], lm[133]
    else:
        upper, lower, corner1, corner2 = lm[386], lm[374], lm[263], lm[362]
    vert = np.linalg.norm(upper - lower)
    hor = np.linalg.norm(corner2 - corner1)
    return vert / hor if hor > 0 else 0.0

def eyebrow_distance(lm):
    return np.linalg.norm(lm[70] - lm[300])

def mouth_corner_slope(lm):
    left, right = lm[78], lm[308]
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    return dy / dx if dx != 0 else 0.0

def detect_emotion_from_frame(frame):
    lm = get_landmarks(frame)
    if lm is None:
        return "neutral"
    
    mar = mouth_aspect_ratio(lm)
    ear = (eye_aspect_ratio(lm, True) + eye_aspect_ratio(lm, False)) / 2.0
    brow_dist = eyebrow_distance(lm)
    slope = mouth_corner_slope(lm)
    face_width = np.linalg.norm(lm[234] - lm[454])
    
    if mar > 0.15:
        return "happy"
    elif ear > 0.25:
        return "surprised"
    elif brow_dist < 0.15 * face_width:
        return "angry"
    elif slope > 0.2:
        return "sad"
    return "neutral"

def detect_emotion_from_text(text: str) -> str:
    text_lower = text.lower()
    emotion_keywords = {
        "happy": ["happy", "joy", "excited", "great", "awesome"],
        "sad": ["sad", "depressed", "down", "upset", "hurt"],
        "angry": ["angry", "mad", "furious", "annoyed", "hate"],
        "frustrated": ["frustrated", "stuck", "annoying", "difficult"],
        "confused": ["confused", "don't understand", "unclear", "what"],
        "excited": ["excited", "can't wait", "amazing", "wow"],
        "anxious": ["worried", "nervous", "anxious", "scared"],
        "tired": ["tired", "exhausted", "sleepy", "drained"]
    }
    
    if text.count('!') >= 2:
        return "excited"
    elif text.count('?') >= 2:
        return "confused"
    elif text.isupper() and len(text) > 10:
        return "angry"
    
    emotion_scores = {emotion: sum(1 for kw in kws if kw in text_lower) 
                     for emotion, kws in emotion_keywords.items()}
    if emotion_scores:
        return max(emotion_scores, key=emotion_scores.get)
    return "neutral"

# Chat state management functions
def load_model_state(selected_model_name):
    global tokenizer, model, device, model_name
    model_name = selected_model_name
    try:
        tokenizer, model, device = load_model(model_name)
        return f"✅ Model loaded on {device}"
    except Exception as e:
        return f"❌ Model loading failed: {str(e)}"

def speak_text(text):
    engine.say(text)
    engine.runAndWait()

def generate_response_state(user_input, emotion, history=None, temperature=0.8, top_k=50, top_p=0.9, max_tokens=50):
    global messages, conversation_history, tokenizer, model, device, model_name
    if not model:
        return "Model not loaded"
    
    if history is None:
        history = conversation_history
    
    prompt = build_prompt(
        history=history,
        user_input=user_input,
        emotion_label=emotion,
        model_type=model_name,
        tokenizer=tokenizer
    )
    
    response = generate_response(
        prompt=prompt,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        history=history,
        user_input=user_input,
        emotion=emotion
    )
    
    is_valid, final_response = validate_response(response, user_input, emotion)
    if not is_valid:
        final_response = get_smart_fallback(user_input, emotion)
    
    # Speak the response in a separate thread
    threading.Thread(target=speak_text, args=(final_response,), daemon=True).start()
    
    conversation_history.append((user_input, final_response))
    messages.append((user_input, final_response, emotion))
    conversation_history = truncate_history(conversation_history, tokenizer, max_tokens=800)
    
    return final_response

def format_chat():
    global messages
    emotion_emoji = {
        "happy": "😊", "sad": "😢", "angry": "😠", "frustrated": "😤",
        "confused": "🤔", "excited": "🤩", "anxious": "😰", "tired": "😴"
    }
    return [(f"{emotion_emoji.get(emotion, '💬')} {user_msg}", bot_msg) 
            for user_msg, bot_msg, emotion in messages]

def clear_chat():
    global messages, conversation_history
    messages = []
    conversation_history = []
    return format_chat()

# Application control functions
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
    global recording, audio_thread
    if not recording:
        return format_chat(), "No active recording", "No transcription", "No response"
    recording = False
    if audio_thread:
        audio_thread.join(timeout=2.0)
        audio_thread = None
    
    audio_path = audio_queue.get()
    emotions = [detect_emotion_from_frame(frame) for frame in snapshots if detect_emotion_from_frame(frame)]
    most_common_emotion = Counter(emotions).most_common(1)[0][0] if emotions else "neutral"
    transcription = transcribe_audio(audio_path)
    
    user_input = f"{most_common_emotion}: {transcription}"
    response = generate_response_state(user_input, most_common_emotion) if model else "Model not loaded"
    
    if audio_path and os.path.exists(audio_path):
        os.remove(audio_path)
    
    # Return the updated chat history along with other outputs
    chat_history = format_chat()
    return chat_history, most_common_emotion, transcription, response

def cleanup():
    global cap
    if cap and cap.isOpened():
        cap.release()
    cap = None
    engine.stop()

# Gradio interface
with gr.Blocks(title="Emotion-Aware Chat App", theme=gr.themes.Soft()) as demo:
    # Model selection
    with gr.Row():
        model_dropdown = gr.Dropdown(choices=list(model_options.keys()), label="Select Model", value="DialoGPT Large (355M)")
        load_btn = gr.Button("Load Model")
        model_status = gr.Textbox(label="Model Status", interactive=False)
    
    # Main layout
    with gr.Row():
        # Chat column (60% width)
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=400, label="Chat")
            user_input = gr.Textbox(lines=1, label="Type your message...", placeholder="Type here and press Enter")
            with gr.Row():
                submit_btn = gr.Button("Send")
                clear_btn = gr.Button("Clear Chat")
                start_video_call_btn = gr.Button("Start Video Call")
            with gr.Accordion("Generation Settings", open=False):
                temperature = gr.Slider(0.6, 1.2, value=0.8, step=0.1, label="Temperature")
                top_k = gr.Slider(30, 100, value=50, step=10, label="Top-k")
                top_p = gr.Slider(0.7, 0.95, value=0.9, step=0.05, label="Top-p")
                max_tokens = gr.Slider(20, 100, value=50, step=10, label="Max tokens")
        
        # Video call column (40% width, hidden by default)
        with gr.Column(scale=2, visible=False) as video_call_col:
            image_out = gr.Image(label="Live Webcam", streaming=True)
            status_out = gr.Textbox(label="Recording Status", value="Not recording", interactive=False)
            with gr.Row():
                start_btn = gr.Button("Start Recording")
                stop_btn = gr.Button("Stop and Process")
                close_video_call_btn = gr.Button("Close Video Call")
            emotion_out = gr.Textbox(label="Detected Emotion", interactive=False)
            transcript_out = gr.Textbox(label="Transcription", interactive=False)
            response_out = gr.Textbox(label="Generated Response", interactive=False)

    # Event handlers
    load_btn.click(
        fn=lambda model_name: load_model_state(model_options[model_name]),
        inputs=model_dropdown,
        outputs=model_status
    )
    
    def text_chat_submit(input_text, temp, tk, tp, mt):
        if not input_text.strip():
            return "", format_chat()
        emotion = detect_emotion_from_text(input_text)
        response = generate_response_state(input_text, emotion, temperature=temp, top_k=tk, top_p=tp, max_tokens=mt)
        return "", format_chat()
    
    submit_btn.click(
        fn=text_chat_submit,
        inputs=[user_input, temperature, top_k, top_p, max_tokens],
        outputs=[user_input, chatbot]
    )
    user_input.submit(
        fn=text_chat_submit,
        inputs=[user_input, temperature, top_k, top_p, max_tokens],
        outputs=[user_input, chatbot]
    )
    
    clear_btn.click(fn=clear_chat, outputs=chatbot)
    
    start_btn.click(fn=start_recording, outputs=status_out)
    
    def process_video_call():
        chat_history, emotion, transcription, response = stop_and_process()
        return chat_history, emotion, transcription, response
    
    stop_btn.click(
        fn=process_video_call,
        outputs=[chatbot, emotion_out, transcript_out, response_out]
    )
    
    start_video_call_btn.click(
        fn=lambda: gr.update(visible=True),
        inputs=None,
        outputs=video_call_col
    )
    
    def close_video_call():
        global recording, audio_thread
        if recording:
            recording = False
            if audio_thread:
                audio_thread.join(timeout=2.0)
                audio_thread = None
        return gr.update(visible=False), "Not recording"
    
    close_video_call_btn.click(
        fn=close_video_call,
        inputs=None,
        outputs=[video_call_col, status_out]
    )
    
    demo.load(video_stream, None, image_out)
    demo.unload(cleanup)

if __name__ == "__main__":
    demo.launch()