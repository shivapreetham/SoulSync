#updated
# Moved audio recording and transcription from main.py.
# Uses a queue for thread-safe communication.
# Cleans up temporary audio files.
# Validates Vosk model path at runtime.

# Added logger from logging_utils.py.
# Added error handling for audio recording and transcription.
# Ensured audio_queue.put(None) on errors to prevent blocking.
# Logged transcription results and file cleanup.
import pyaudio
import soundfile as sf
import numpy as np
from vosk import Model, KaldiRecognizer
import json
import queue
import os
from typing import Optional, Callable
from logging_utils import setup_logger

logger = setup_logger("audio_utils")
VOSK_MODEL_PATH = r"D:\College_Life\projects\Hackathon\Edge AI\SoulSync _ main\SoulSync\vosk-model-small-en-us-0.15"#"vosk-model-en-us-0.42-gigaspeech"

def record_audio(audio_queue: queue.Queue, is_recording: Callable[[], bool]) -> None:
    """Record audio and save to a temporary file until recording stops."""
    if not os.path.isdir(VOSK_MODEL_PATH):
        logger.error(f"Vosk model not found at {VOSK_MODEL_PATH}")
        audio_queue.put(None)
        return
    
    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
        logger.info("Audio stream opened")
        frames = []
        
        while is_recording():
            try:
                data = stream.read(1024, exception_on_overflow=False)
                frames.append(data)
            except Exception as e:
                logger.error(f"Audio read error: {str(e)}")
                break
        
        stream.stop_stream()
        stream.close()
        logger.info("Audio stream closed")
        
        if frames:
            audio_data = b''.join(frames)
            audio_path = "temp_audio.wav"
            sf.write(audio_path, np.frombuffer(audio_data, dtype=np.int16), 16000)
            audio_queue.put(audio_path)
            logger.info(f"Audio recorded and saved to {audio_path}")
        else:
            logger.warning("No audio frames recorded")
            audio_queue.put(None)
    except Exception as e:
        logger.error(f"Audio recording setup error: {str(e)}")
        audio_queue.put(None)
    finally:
        p.terminate()

def transcribe_audio(audio_path: Optional[str]) -> str:
    """Transcribe audio using Vosk."""
    if not audio_path or not os.path.exists(audio_path):
        logger.warning("No audio file provided or file does not exist")
        return "No audio recorded"
    
    try:
        vosk_model = Model(VOSK_MODEL_PATH)
        data, samplerate = sf.read(audio_path)
        if data.ndim > 1:
            data = data[:, 0]
        if data.dtype != np.int16:
            data = (data * 32767).astype(np.int16)
        recognizer = KaldiRecognizer(vosk_model, samplerate)
        recognizer.AcceptWaveform(data.tobytes())
        result = json.loads(recognizer.FinalResult())
        transcription = result.get("text", "No speech detected")
        logger.info(f"Transcription: {transcription}")
        return transcription
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        return f"Transcription error: {str(e)}"
    finally:
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                logger.info(f"Removed temporary audio file: {audio_path}")
            except Exception as e:
                logger.error(f"Failed to remove audio file: {str(e)}")