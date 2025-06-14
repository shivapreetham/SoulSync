import cv2
import mediapipe as mp
import numpy as np
import time
import os

# Create output directory if it doesn't exist
os.makedirs('debug_images', exist_ok=True)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configure FaceMesh with appropriate parameters
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,      # Process as video stream
    max_num_faces=1,              # Detect one face
    refine_landmarks=True,        # Refine landmarks for eyes and lips
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Function to find a working camera
def find_working_camera(max_index=5):
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                print(f"Debug: Camera found at index {index}")
                return cap, index
        cap.release()
    return None, None

# Find a working camera
cap, camera_index = find_working_camera()
if cap is None or camera_index is None:
    print("Error: No working camera found. Check your camera connection or permissions.")
    exit()

print(f"Using camera index: {camera_index}")
print("Camera warming up...")

# Allow camera to warm up
time.sleep(2)

# Capture and process frames
detection_attempts = 0
max_attempts = 30  # Try up to 30 frames
face_detected = False

while detection_attempts < max_attempts and not face_detected:
    ret, frame = cap.read()
    if not ret or frame is None:
        print(f"Attempt {detection_attempts+1}: Failed to capture frame")
        detection_attempts += 1
        time.sleep(0.1)
        continue
    
    # Flip frame horizontally for natural selfie-view
    frame = cv2.flip(frame, 1)
    
    # Convert to RGB for FaceMesh
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Save original frame for debugging
    cv2.imwrite(f'debug_images/original_{detection_attempts}.jpg', frame)
    print(f"Saved original frame {detection_attempts}")
    
    # Process with FaceMesh
    results = face_mesh.process(frame_rgb)
    
    # Create a copy for drawing landmarks
    output_frame = frame.copy()
    
    if results.multi_face_landmarks:
        print(f"Face detected in frame {detection_attempts}!")
        face_detected = True
        
        # Draw all detected faces
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=output_frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
        
        # Save output frame with landmarks
        cv2.imwrite('debug_images/face_detected.jpg', output_frame)
        print("Saved face detection image with landmarks")
        
        # Display the image with landmarks
        cv2.imshow('Face Detected', output_frame)
        cv2.waitKey(3000)  # Show for 3 seconds
    else:
        print(f"Attempt {detection_attempts+1}: No face detected")
        
        # Save frame where face wasn't detected
        cv2.imwrite(f'debug_images/no_face_{detection_attempts}.jpg', frame)
    
    detection_attempts += 1
    time.sleep(0.2)  # Short delay between captures

# Clean up
cap.release()
cv2.destroyAllWindows()

if not face_detected:
    print(f"Error: Face not detected after {max_attempts} attempts.")
    print("Possible reasons:")
    print("1. Poor lighting - ensure your face is well-lit")
    print("2. Face too far from camera - move closer")
    print("3. Camera angle - face the camera directly")
    print("4. Obstructions - remove glasses/mask if possible")
    print("Check debug_images folder for captured frames")
else:
    print("Face detection successful!")