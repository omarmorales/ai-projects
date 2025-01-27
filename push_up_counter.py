import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(1)

# Variables for push-up counting
push_up_count = 0
push_up_position = "up"  # Start in the "up" position
cooldown_frames = 30  # Number of frames to wait before counting another push-up
cooldown_counter = 0  # Counter for cooldown period

# Custom color for the counter (in BGR format)
counter_color = (255, 0, 0)  # Red: (B=0, G=0, R=255)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Process the image and detect pose
    results = pose.process(image)

    # Convert the image back to BGR for rendering
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        # Draw pose landmarks
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        # Get landmarks for shoulders and elbows
        landmarks = results.pose_landmarks.landmark
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]

        # Calculate average shoulder and elbow y-coordinates
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        elbow_y = (left_elbow.y + right_elbow.y) / 2

        # Push-up logic
        if cooldown_counter > 0:
            cooldown_counter -= 1  # Decrease cooldown counter
        else:
            if shoulder_y < elbow_y and push_up_position == "up":
                push_up_position = "down"  # Transition to "down" position
            elif shoulder_y > elbow_y and push_up_position == "down":
                push_up_position = "up"  # Transition to "up" position
                push_up_count += 1  # Increment push-up count
                cooldown_counter = cooldown_frames  # Start cooldown period

        # Display push-up count with custom color
        cv2.putText(
            image,
            f"Numero de lagartijas: {push_up_count}",
            (10, 50),  # Position of the text (top-left corner)
            cv2.FONT_HERSHEY_SIMPLEX,  # Font type
            1,  # Font scale
            counter_color,  # Text color (BGR format)
            2,  # Thickness of the text
            cv2.LINE_AA,  # Line type
        )

    # Display the frame
    cv2.imshow("Push-Up Counter", image)

    # Exit on 'q' key press
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()