import gradio as gr
import torch
import time
import numpy as np
from llm_utils import load_model, generate_response, get_smart_fallback
from chat_utils import build_prompt, truncate_history, validate_response, detect_conversation_quality
import onnxruntime
import requests
import os
import io
import json
from scipy.io import wavfile

# Emotion detection function
def detect_emotion_from_text(text: str) -> str:
    text_lower = text.lower()
    emotion_keywords = {
        "happy": ["happy", "joy", "excited", "great", "awesome", "wonderful", "amazing", "love", "fantastic"],
        "sad": ["sad", "depressed", "down", "upset", "hurt", "disappointed", "cry", "awful", "terrible"],
        "angry": ["angry", "mad", "furious", "annoyed", "irritated", "hate", "frustrated", "pissed"],
        "frustrated": ["frustrated", "stuck", "annoying", "difficult", "struggling", "can't", "won't work"],
        "confused": ["confused", "don't understand", "unclear", "puzzled", "lost", "what", "how", "why"],
        "excited": ["excited", "can't wait", "amazing", "incredible", "wow", "omg", "awesome", "fantastic"],
        "anxious": ["worried", "nervous", "anxious", "scared", "afraid", "concerned", "stress", "panic"],
        "tired": ["tired", "exhausted", "sleepy", "worn out", "drained", "fatigue"]
    }
    
    if text.count('!') >= 2:
        return "excited"
    elif text.count('?') >= 2:
        return "confused"
    elif text.isupper() and len(text) > 10:
        return "angry"
    
    emotion_scores = {}
    for emotion, keywords in emotion_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score > 0:
            emotion_scores[emotion] = score
    
    if emotion_scores:
        return max(emotion_scores, key=emotion_scores.get)
    
    return "neutral"

# Model options dictionary
model_options = {
    "DialoGPT Large (355M)": "microsoft/DialoGPT-large",
    "BlenderBot 3B": "facebook/blenderbot-3B",
    "BlenderBot 1B": "facebook/blenderbot-1B-distill",
    "GPT-2 Large (774M)": "gpt2-large",
    "GPT-2 XL (1.5B)": "gpt2-xl"
}

# Piper TTS Implementation
class PiperTTS:
    def __init__(self):
        self.session = None
        self.available = False
        self.sample_rate = 22050
        self.model_path = "en_US-lessac-medium.onnx"
        self.config_path = "en_US-lessac-medium.onnx.json"
        
        # Download model and config if needed
        self.download_model()
        
        try:
            # Initialize ONNX runtime session
            self.session = onnxruntime.InferenceSession(
                self.model_path,
                providers=["CPUExecutionProvider"]
            )
            self.available = True
            print("Piper TTS initialized successfully")
        except Exception as e:
            print(f"Piper TTS initialization failed: {e}")
            self.available = False
    
    def download_model(self):
        base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/"
        files = {
            self.model_path: base_url + "en_US-lessac-medium.onnx",
            self.config_path: base_url + "en_US-lessac-medium.onnx.json"
        }
        
        for filename, url in files.items():
            if not os.path.exists(filename):
                print(f"Downloading {filename}...")
                try:
                    response = requests.get(url)
                    with open(filename, "wb") as f:
                        f.write(response.content)
                    print(f"Downloaded {filename}")
                except Exception as e:
                    print(f"Error downloading {filename}: {e}")
    
    def text_to_audio(self, text: str, emotion: str) -> np.ndarray:
        """Convert text to audio waveform using Piper TTS"""
        if not self.available or not self.session:
            return None
            
        try:
            # Emotion-based speed adjustments
            emotion_speeds = {
                "excited": 1.3,
                "angry": 1.2,
                "sad": 0.7,
                "tired": 0.6,
                "confused": 0.8,
                "anxious": 1.1
            }
            speed = emotion_speeds.get(emotion, 1.0)
            
            # Tokenize and convert to ids
            from piper_phonemize import phonemize_espeak
            phonemes = phonemize_espeak(text, "en-us")
            phoneme_ids = [self.config["phoneme_id_map"][p] for p in phonemes]
            
            # Create input tensors
            phoneme_ids_array = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
            phoneme_ids_lengths = np.array([phoneme_ids_array.shape[1]], dtype=np.int64)
            scales = np.array([
                speed,  # length scale (speed)
                1.0,    # noise scale
                0.667,  # noise_w scale
            ], dtype=np.float32)
            
            # Run inference
            audio = self.session.run(
                None,
                {
                    "input": phoneme_ids_array,
                    "input_lengths": phoneme_ids_lengths,
                    "scales": scales,
                }
            )[0].squeeze()
            
            return audio.astype(np.int16)
        except Exception as e:
            print(f"TTS generation error: {e}")
            return None

# Initialize TTS system
try:
    # Try to import phonemizer
    from piper_phonemize import phonemize_espeak
    tts_engine = PiperTTS()
except ImportError:
    print("Phonemizer not installed. TTS disabled.")
    tts_engine = type('DummyTTS', (), {'available': False, 'text_to_audio': lambda *args: None})()

# State management class
class ChatState:
    def __init__(self):
        self.messages = []
        self.conversation_history = []
        self.quality_metrics = {}
        self.tokenizer = None
        self.model = None
        self.device = None
        self.model_name = list(model_options.values())[0]
        self.tts_enabled = True
        self.load_model()
    
    def load_model(self, model_name=None):
        if model_name:
            self.model_name = model_name
        try:
            self.tokenizer, self.model, self.device = load_model(self.model_name)
            return f"✅ Model loaded on {self.device}"
        except Exception as e:
            return f"❌ Model loading failed: {str(e)}"
    
    def clear_chat(self):
        self.messages = []
        self.conversation_history = []
        self.quality_metrics = {}
        return self.format_chat(), self.get_stats()
    
    def format_chat(self):
        formatted = []
        emotion_emoji = {
            "happy": "😊", "sad": "😢", "angry": "😠", "frustrated": "😤",
            "confused": "🤔", "excited": "🤩", "anxious": "😰", "tired": "😴"
        }
        
        for user_msg, bot_msg, emotion in self.messages:
            emoji = emotion_emoji.get(emotion, "💬")
            formatted.append((f"{emoji} {user_msg}", bot_msg))
        return formatted
    
    def get_stats(self):
        stats = {}
        stats["Messages"] = len(self.messages)
        
        if self.messages:
            emotions = [msg[2] for msg in self.messages]
            most_common = max(set(emotions), key=emotions.count) if emotions else "neutral"
            stats["Dominant Emotion"] = most_common.title()
            
            if self.quality_metrics:
                quality = self.quality_metrics.get("quality", "unknown")
                stats["Conversation Quality"] = quality.title()
        
        return stats
    
    def generate_response(self, user_input, temperature, top_k, top_p, max_tokens):
        if not self.model:
            return "Model not loaded. Please select a model first.", self.format_chat(), self.get_stats(), None
        
        emotion = detect_emotion_from_text(user_input)
        
        prompt = build_prompt(
            history=self.conversation_history,
            user_input=user_input,
            emotion_label=emotion,
            model_type=self.model_name,
            tokenizer=self.tokenizer
        )
        
        response = generate_response(
            prompt=prompt,
            tokenizer=self.tokenizer,
            model=self.model,
            device=self.device,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            history=self.conversation_history,
            user_input=user_input,
            emotion=emotion
        )
        
        is_valid, final_response = validate_response(response, user_input, emotion)
        if not is_valid:
            final_response = get_smart_fallback(user_input, emotion)
        
        # Generate audio if TTS is enabled
        audio_data = None
        sample_rate = 22050  # Default sample rate for Piper
        
        if self.tts_enabled and tts_engine.available:
            audio_data = tts_engine.text_to_audio(final_response, emotion)
        
        self.conversation_history.append((user_input, final_response))
        self.messages.append((user_input, final_response, emotion))
        
        self.conversation_history = truncate_history(
            self.conversation_history, 
            self.tokenizer,
            max_tokens=800
        )
        
        self.quality_metrics = detect_conversation_quality(
            self.conversation_history
        )
        
        return "", self.format_chat(), self.get_stats(), (sample_rate, audio_data) if audio_data is not None else None

# Create initial state
state = ChatState()

# UI Components
with gr.Blocks(title="Soul Sync Chat", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Soul Sync Chat")
    gr.Markdown("*An advanced emotion-aware chatbot for meaningful conversations*")
    
    with gr.Row():
        # Chat Interface
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500, label="Conversation")
            user_input = gr.Textbox(label="Type your message here...", placeholder="Type your message and press Enter")
            submit_btn = gr.Button("Send")
            
            # Hidden audio component for playback
            audio_output = gr.Audio(visible=False, autoplay=True)
        
        # Sidebar
        with gr.Column(scale=1):
            with gr.Accordion("🎛️ Model Configuration", open=True):
                model_dropdown = gr.Dropdown(
                    choices=list(model_options.keys()),
                    value="DialoGPT Large (355M)",
                    label="Choose Model",
                    info="Larger models provide better conversations but need more memory"
                )
                model_status = gr.Textbox(label="Model Status", interactive=False)
                load_btn = gr.Button("Load Selected Model")
                
                if torch.cuda.is_available():
                    gpu_info = gr.Textbox(label="GPU Memory", interactive=False, value="GPU detected")
            
            with gr.Accordion("🔊 Voice Settings", open=True):
                tts_toggle = gr.Checkbox(
                    label="Enable Voice Responses", 
                    value=True,
                    info="Chatbot will speak its responses"
                )
                if not tts_engine.available:
                    gr.Warning("TTS engine not available. Audio features disabled.")
            
            with gr.Accordion("⚙️ Generation Settings", open=False):
                temperature = gr.Slider(0.6, 1.2, value=0.8, step=0.1, label="Temperature", 
                                      info="Higher = more creative, Lower = more focused")
                top_k = gr.Slider(30, 100, value=50, step=10, label="Top-k", 
                                 info="Number of top tokens to consider")
                top_p = gr.Slider(0.7, 0.95, value=0.9, step=0.05, label="Top-p", 
                                 info="Nucleus sampling threshold")
                max_tokens = gr.Slider(20, 100, value=50, step=10, label="Max tokens", 
                                     info="Maximum response length")
            
            with gr.Accordion("📊 Session Stats", open=False):
                stats = gr.JSON(label="Statistics")
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear Chat")
                export_btn = gr.Button("💾 Export Chat")
            
            gr.Markdown("---")
            gr.Markdown("**Soul Sync v1.0**")
            gr.Markdown("Emotion-aware conversational AI")
    
    # Event handlers
    def load_model_wrapper(model_name):
        model_key = model_options[model_name]
        status = state.load_model(model_key)
        return status
    
    def generate_wrapper(input_text, temperature, top_k, top_p, max_tokens):
        if not input_text.strip():
            return "", state.format_chat(), state.get_stats(), None
        return state.generate_response(input_text, temperature, top_k, top_p, max_tokens)
    
    def export_chat():
        if not state.messages:
            return None
        chat_text = "\n".join([
            f"User: {msg[0]}\nBot: {msg[1]}\nEmotion: {msg[2]}\n---"
            for msg in state.messages
        ])
        return gr.File(value=chat_text.encode('utf-8'), filename="soul_sync_chat.txt")
    
    def toggle_tts(enabled):
        state.tts_enabled = enabled
        return state.format_chat(), state.get_stats()
    
    # Initial state setup
    demo.load(
        fn=lambda: (state.format_chat(), state.get_stats(), state.load_model()),
        inputs=None,
        outputs=[chatbot, stats, model_status]
    )
    
    # Connect components
    load_btn.click(
        fn=load_model_wrapper,
        inputs=model_dropdown,
        outputs=model_status
    )
    
    submit_btn.click(
        fn=generate_wrapper,
        inputs=[user_input, temperature, top_k, top_p, max_tokens],
        outputs=[user_input, chatbot, stats, audio_output]
    )
    
    user_input.submit(
        fn=generate_wrapper,
        inputs=[user_input, temperature, top_k, top_p, max_tokens],
        outputs=[user_input, chatbot, stats, audio_output]
    )
    
    clear_btn.click(
        fn=state.clear_chat,
        inputs=None,
        outputs=[chatbot, stats]
    )
    
    export_btn.click(
        fn=export_chat,
        inputs=None,
        outputs=gr.File()
    )
    
    tts_toggle.change(
        fn=toggle_tts,
        inputs=tts_toggle,
        outputs=[chatbot, stats]
    )

if __name__ == "__main__":
    # Use a lightweight server configuration
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        favicon_path=None,
        inbrowser=True
    )