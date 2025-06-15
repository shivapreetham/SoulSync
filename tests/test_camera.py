# test_camera.py
import cv2
import time

def test_camera(index, backend):
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        print(f"Failed to open camera at index {index} with backend {backend}")
        return
    time.sleep(1)  # Warm-up
    for _ in range(30):
        ret, frame = cap.read()
        if ret and frame is not None:
            cv2.imshow(f"Camera {index} Backend {backend}", frame)
        else:
            print(f"Failed to read frame at index {index} with backend {backend}")
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
    for index in range(3):
        print(f"Testing index {index} with backend {backend}")
        test_camera(index, backend)