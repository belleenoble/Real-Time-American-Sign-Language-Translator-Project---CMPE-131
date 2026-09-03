import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam opened successfully. Press Q to quit.")

while True:
    success, frame = camera.read()

    if not success:
        print("Error: Could not read webcam frame.")
        break

    # Mirror the image like a normal webcam
    frame = cv2.flip(frame, 1)

    cv2.imshow("ASL Recognition - Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()