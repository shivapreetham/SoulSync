# tests/test_main.py
from main import stop_and_process, recording, snapshots, model
from audio_utils import transcribe_audio

def test_stop_and_process():
    global recording, snapshots, model
    recording = True
    snapshots = [np.zeros((480, 640, 3), dtype=np.uint8)]
    model = True  # Mock model
    emotion, transcription, response = stop_and_process()
    assert emotion in ["neutral", "happy", "sad", "angry", "surprised"]
    assert isinstance(transcription, str)
    assert isinstance(response, str)