import os
import cv2 as cv
import mediapipe as mp

class HandTracking:
    def __init__(self):
        # Initialize mediapipe hands module
        self.mphands = mp.solutions.hands
        self.mpdrawing = mp.solutions.drawing_utils

        # Set the desired window width and height
        self.winwidth = 640
        self.winheight = 480

        # Initialize video capture
        self.vidcap = cv.VideoCapture(0)

    def track(self):
        # Initialize hand tracking
        with self.mphands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while self.vidcap.isOpened():
                ret, frame = self.vidcap.read()
                if not ret:
                    break

                # Convert the BGR image to RGB
                rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

                # Process the frame for hand tracking
                processFrames = hands.process(rgb_frame)

                # Draw landmarks on the frame
                if processFrames.multi_hand_landmarks:
                    for lm in processFrames.multi_hand_landmarks:
                        self.mpdrawing.draw_landmarks(frame, lm, self.mphands.HAND_CONNECTIONS)

                # Resize the frame to the desired window size
                resized_frame = cv.resize(frame, (self.winwidth, self.winheight))

                # Display the resized frame
                cv.imshow('Hand Tracking', resized_frame)

                # Exit loop by pressing 'q'
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

            # Release the video capture and close windows
            self.vidcap.release()
            cv.destroyAllWindows()

if __name__ == '__main__':
    handTracking = HandTracking()
    handTracking.track()
