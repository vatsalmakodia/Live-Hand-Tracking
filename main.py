import cv2
import mediapipe as mp

# storing required variables to use
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()                            # we can change max hands in Hands class
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
while True :
    ret,img = cap.read()                            # cap live and store
    
    # storing and printing landmarks
    results = hands.process(img)
    print(results.multi_hand_landmarks)
    
    # for detecting hand landmarks and displaying
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec)  # open this class for selecting colour

    cv2.imshow("feed", img)
    key = cv2.waitKey(1)                             # wait till any key press
    if key == ord("q"):                              # exit loop on 'q' key press
        break

cap.release()                                        # release video capture object
cv2.destroyAllWindows()