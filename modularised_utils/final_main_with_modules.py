#updated
# Imported new modules and used their functions.
# Moved model_options to config_utils.py.
# Added logging for key operations.
# Simplified global state management.
# Kept Gradio interface intact but updated to use new modules.

# Added error handling in start_recording and stop_and_process.
# Added 5-second timeout for audio_queue.get() to prevent blocking.
# Cleared audio_queue in stop_and_process to avoid stale data.
# Logged snapshot count and errors.
# Added error handling in close_video_call and cleanup.
import gradio as gr
import threading
import queue
import time
import pyttsx3
from llm_utils import load_model, generate_response, get_smart_fallback
from chat_utils import build_prompt, truncate_history, validate_response, detect_conversation_quality
from emotion_utils import detect_emotion_from_text, aggregate_emotions
from video_utils import video_stream, process_video_frame, find_available_camera
from audio_utils import record_audio, transcribe_audio
from config_utils import get_model_options
from logging_utils import setup_logger

# Initialize logger
logger = setup_logger("soulsync")

# Initialize TTS engine with thread-safe handling
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)
tts_lock = threading.Lock()

# Global state
messages = []
conversation_history = []
tokenizer = None
model = None
device = None
model_name = None
recording = False
snapshots = []
audio_queue = queue.Queue()
cap = None
last_snapshot_time = 0
stream_active = True

def load_model_state(selected_model_name: str) -> tuple[str, bool]:
    """Load the selected model and enable processing."""
    global tokenizer, model, device, model_name
    model_options = get_model_options()
    model_name = selected_model_name
    try:
        tokenizer, model, device = load_model(model_options[model_name])
        logger.info(f"Model {model_name} loaded on {device}")
        return f"✅ Model loaded on {device}", True  # Enable stop button
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        return f"❌ Model loading failed: {str(e)}", False

def speak_text(text: str) -> None:
    """Speak text using TTS with thread-safe locking."""
    try:
        with tts_lock:
            engine.say(text)
            engine.runAndWait()
    except Exception as e:
        logger.error(f"TTS error: {str(e)}")

def generate_response_state(
    user_input: str,
    emotion: str,
    history: list = None,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    max_tokens: int = 50
) -> str:
    """Generate a response based on user input and emotion."""
    global messages, conversation_history, tokenizer, model, device, model_name
    if not model:
        logger.warning("Model not loaded")
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
    
    threading.Thread(target=speak_text, args=(final_response,), daemon=True).start()
    
    conversation_history.append((user_input, final_response))
    messages.append((user_input, final_response, emotion))
    conversation_history = truncate_history(conversation_history, tokenizer, max_tokens=800)
    
    quality_metrics = detect_conversation_quality(conversation_history)
    logger.info(f"Conversation quality: {quality_metrics['quality']} | Metrics: {quality_metrics['metrics']} | Suggestions: {quality_metrics['suggestions']}")
    
    logger.info(f"Generated response for input: {user_input[:50]}... | Emotion: {emotion}")
    return final_response

def format_chat() -> list:
    """Format chat history for display in Gradio Chatbot."""
    emotion_emoji = {
        "happy": "😊", "sad": "😢", "angry": "😠", "frustrated": "😤",
        "confused": "🤔", "excited": "🤩", "anxious": "😰", "tired": "😴"
    }
    chat_messages = []
    for user_msg, bot_msg, emotion in messages:
        chat_messages.append({"role": "user", "content": f"{emotion_emoji.get(emotion, '💬')} {user_msg}"})
        chat_messages.append({"role": "assistant", "content": bot_msg})
    return chat_messages

def clear_chat() -> list:
    """Clear chat history."""
    global messages, conversation_history
    messages = []
    conversation_history = []
    logger.info("Chat history cleared")
    return []

def start_recording() -> tuple[str, str]:
    """Start recording audio and video."""
    global recording, snapshots, cap, last_snapshot_time, stream_active
    if recording:
        return "Already recording", ""
    try:
        recording = True
        stream_active = True
        snapshots.clear()
        last_snapshot_time = time.time()
        if cap is not None:
            cap.release()
        cap = find_available_camera()
        if cap is None:
            recording = False
            logger.warning("No camera found")
            return "No camera found", ""
        
        threading.Thread(target=record_audio, args=(audio_queue, lambda: recording), daemon=True).start()
        logger.info("Started recording audio and video")
        warning = "Please load a model before stopping the recording." if model is None else ""
        return "Recording audio and video... Speak clearly for 10–15 seconds.", warning
    except Exception as e:
        recording = False
        logger.error(f"Start recording error: {str(e)}")
        return f"Failed to start recording: {str(e)}", ""

def stop_and_process() -> tuple[str, str, str]:
    """Stop recording and process audio/video."""
    global recording, snapshots, cap, last_snapshot_time, stream_active
    if not recording:
        return "No active recording", "No transcription", "No response"
    
    try:
        recording = False
        stream_active = False
        if cap:
            cap.release()
            cap = None
        
        try:
            audio_path = audio_queue.get(timeout=10.0)
        except queue.Empty:
            logger.error("Audio queue timed out")
            return "Processing error", "No audio recorded", "No response"
        
        logger.info(f"Processing {len(snapshots)} snapshots")
        emotion = aggregate_emotions(snapshots) if snapshots else "neutral"
        transcription = transcribe_audio(audio_path)
        response = generate_response_state(transcription, emotion) if model and transcription and transcription != "No speech detected" else "Model not loaded or no transcription"
        logger.info(f"Response generated: {response}")
        
        snapshots.clear()
        last_snapshot_time = 0
        logger.info(f"Processed recording | Emotion: {emotion} | Transcription: {transcription[:50]}...")
        return emotion, transcription, response
    except Exception as e:
        logger.error(f"Stop and process error: {str(e)}")
        return "Processing error", f"Error: {str(e)}", "No response"
    finally:
        while not audio_queue.empty():
            audio_queue.get()

def cleanup() -> None:
    """Clean up resources."""
    global cap, recording, stream_active
    try:
        if cap:
            cap.release()
            cap = None
        recording = False
        stream_active = False
        with tts_lock:
            engine.stop()
        logger.info("Resources cleaned up")
    except Exception as e:
        logger.error(f"Cleanup error: {str(e)}")

# Gradio interface
with gr.Blocks(title="Emotion-Aware Chat App", theme=gr.themes.Soft()) as demo:
    model_options = get_model_options()
    with gr.Row():
        model_dropdown = gr.Dropdown(choices=list(model_options.keys()), label="Select Model", value="DialoGPT Large (355M)")
        load_btn = gr.Button("Load Model")
        model_status = gr.Textbox(label="Model Status", interactive=False)
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=400, label="Chat", type="messages")
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
        
        with gr.Column(scale=2, visible=False) as video_call_col:
            image_out = gr.Image(label="Live Webcam", streaming=True)
            status_out = gr.Textbox(label="Recording Status", value="Not recording", interactive=False)
            warning_out = gr.Textbox(label="Warnings", interactive=False)
            start_btn = gr.Button("Start Recording")
            stop_btn = gr.Button("Stop and Process", interactive=False)
            emotion_out = gr.Textbox(label="Detected Emotion", interactive=False)
            transcript_out = gr.Textbox(label="Transcription", interactive=False)
            response_out = gr.Textbox(label="Generated Response", interactive=False)
            close_video_call_btn = gr.Button("Close Video Call")

    load_btn.click(
        fn=load_model_state,
        inputs=model_dropdown,
        outputs=[model_status, stop_btn]
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
    
    start_btn.click(
        fn=start_recording,
        outputs=[status_out, warning_out]
    )
    stop_btn.click(
        fn=stop_and_process,
        outputs=[emotion_out, transcript_out, response_out]
    )
    
    start_video_call_btn.click(
        fn=lambda: gr.update(visible=True),
        inputs=None,
        outputs=video_call_col
    )
    
    def close_video_call():
        global recording, cap, stream_active
        try:
            recording = False
            stream_active = False
            if cap:
                cap.release()
                cap = None
            return gr.update(visible=False), "Not recording", ""
        except Exception as e:
            logger.error(f"Close video call error: {str(e)}")
            return gr.update(visible=False), f"Error closing video call: {str(e)}", ""
    
    close_video_call_btn.click(
        fn=close_video_call,
        inputs=None,
        outputs=[video_call_col, status_out, warning_out]
    )
    
    def stream_with_globals():
        global stream_active
        start_time = time.time()
        timeout = 60  # Timeout after 60 seconds
        for frame in video_stream(snapshots, recording, last_snapshot_time):
            if not stream_active or time.time() - start_time > timeout:
                logger.info("Stopping video stream")
                break
            yield frame
    
    demo.load(fn=stream_with_globals, inputs=None, outputs=image_out)
    logger.info("Video stream loaded")

demo.unload(cleanup)

if __name__ == "__main__":
    demo.launch()