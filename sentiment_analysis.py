from fer import FER
import cv2

# Initialize the FER detector
detector = FER()

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Detect emotions
    emotions = detector.detect_emotions(frame)

    # Display the emotions on the frame
    for face in emotions:
        x, y, w, h = face['box']
        emotion = max(face['emotions'], key=face['emotions'].get)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow('Video Feed', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()