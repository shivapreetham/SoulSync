import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

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