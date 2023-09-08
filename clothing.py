from math import sqrt
from cvzone.HandTrackingModule import HandDetector
from cvzone.PoseModule import PoseDetector
import cvzone   # we use media pipe model posnet to detect body parts
import cv2      # opencv
import os

# Screen Parameters Setting
cap = cv2.VideoCapture(0)
cap.set(3, 1240)
cap.set(4, 680)
 
# import posenet & hand tracking model 
hand_detector = HandDetector(detectionCon=0.8)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = hand_detector.findHands(img, flipType=False, draw=True)
    if hands:
        hand = hands[0]
        lmList = hand['lmList']
        bbox = hand['bbox']

        # calculate the distance between index and middle fingers
        if len(lmList) >= 2:
            x1, y1 = lmList[8][:2]
            x2, y2 = lmList[12][:2]
            # calculate the distance between the two fingers using the bounding box size as a reference
            bbox_width, bbox_height = bbox[2], bbox[3]
            bbox_diagonal = sqrt(bbox_width ** 2 + bbox_height ** 2)
            dist_pixels = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            dist_inches = (2.5 / bbox_diagonal) * dist_pixels  # assuming average finger length is 2.5 inches
            length, info, img = hand_detector.findDistance(lmList[8][:2], lmList[12][:2], img)
            print(length)
            dist_len = (2.5 / bbox_diagonal) * length
            # print(f"Distance between fingers: {dist_inches:.2f} inches")
            print(f"Distance between fingers: {dist_len:.2f} length")

        # draw the hand landmarks and bounding box
        # img = hand_detector.drawHands(img)
        img = cv2.rectangle(img, bbox, (255, 0, 255), 2)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
