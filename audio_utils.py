import pyaudio
import soundfile as sf
import numpy as np
import os
import queue
import json
import threading
from vosk import Model, KaldiRecognizer

class AudioProcessor:
    def __init__(self, model_path="vosk-model-en-us-0.42-gigaspeech"):
        if not os.path.isdir(model_path):
            raise RuntimeError(f"Model not found at {model_path}")
        self.model = Model(model_path)
        self.audio_queue = queue.Queue()
        self.recording = False
        self.audio_thread = None
        self.audio_path = None

    def record_audio(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, 
                       channels=1, 
                       rate=16000, 
                       input=True, 
                       frames_per_buffer=1024)
        frames = []
        
        while self.recording:
            try:
                data = stream.read(1024, exception_on_overflow=False)
                frames.append(data)
            except OSError as e:
                continue
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        if frames:
            audio_data = b''.join(frames)
            self.audio_path = "temp_audio.wav"
            sf.write(self.audio_path, np.frombuffer(audio_data, dtype=np.int16), 16000)
            self.audio_queue.put(self.audio_path)
        else:
            self.audio_queue.put(None)

    def start_recording(self):
        if not self.recording:
            self.recording = True
            self.audio_thread = threading.Thread(target=self.record_audio, daemon=True)
            self.audio_thread.start()

    def stop_recording(self):
        self.recording = False
        if self.audio_thread:
            self.audio_thread.join(timeout=2.0)
        try:
            return self.audio_queue.get(timeout=3.0)
        except queue.Empty:
            return None

    def transcribe_audio(self, audio_path):
        if not audio_path or not os.path.exists(audio_path):
            return "No audio recorded"
            
        try:
            data, samplerate = sf.read(audio_path)
            if data.ndim > 1:
                data = data[:, 0]
            if data.dtype != np.int16:
                data = (data * 32767).astype(np.int16)
                
            recognizer = KaldiRecognizer(self.model, samplerate)
            recognizer.AcceptWaveform(data.tobytes())
            result = recognizer.FinalResult()
            return json.loads(result).get("text", "No speech detected")
        except Exception as e:
            return f"Transcription error: {str(e)}"

    def cleanup(self):
        if self.audio_path and os.path.exists(self.audio_path):
            os.remove(self.audio_path)