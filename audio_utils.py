import pyaudio
import soundfile as sf
import os
import numpy as np
from vosk import Model, KaldiRecognizer
import json

VOSK_MODEL_PATH = "vosk-model-en-us-0.42-gigaspeech"
if not os.path.isdir(VOSK_MODEL_PATH):
    raise RuntimeError(f"Vosk model not found at {VOSK_MODEL_PATH}")
vosk_model = Model(VOSK_MODEL_PATH)

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