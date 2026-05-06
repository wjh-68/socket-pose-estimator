import cv2
import numpy as np
# ======== Sensor Config =========
CAMERA_ID = 0

def main():

    # Camera
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)
    while True:
        ret, img = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        img_float = img.astype(np.float32)
        img_bright = img_float -50

        # 限制范围并转回 uint8
        img_bright = np.clip(
            img_bright, 0, 255).astype(np.uint8)
        cv2.imshow('image',img)
        cv2.waitKey(1)
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()