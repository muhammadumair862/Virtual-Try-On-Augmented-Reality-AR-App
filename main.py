from cvzone.HandTrackingModule import HandDetector
from cvzone.PoseModule import PoseDetector
import cvzone   # we use media pipe model posnet to detect body parts
import cv2      # opencv
import os
from math import sqrt
import time
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5052)

# Define font settings
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_color = (0, 0, 255)
thickness = 2

# Define camera parameters
distance_from_camera = 95 # inches
image_width_in_pixels = 640 # pixels

# Define body height in inches
body_height = 68 # inches

# Screen Parameters Setting
cap = cv2.VideoCapture(0)
cap.set(3, 1240)
cap.set(4, 680)
 
# import posenet & hand tracking model 
hand_detector = HandDetector(detectionCon=0.8)
pose_detector = PoseDetector()

# image Main Frame 
# main_img_path = "Images"
shirtFolderPath = "./Resources/Shirts"
frameFolderPath = "./Resources/Images"
buttonFolderPath = "./Resources"
listShirts = os.listdir(shirtFolderPath)
listframes = os.listdir(frameFolderPath)
fixedRatio = 262 / 190              # width of Shirt / width of Point 11-12
aspect_ratio = 581 / 440   # aspect ratio
imageNumber = 0
back_imageNumber = 0
scaling_factor = 31 / 1280
# print(listShirts)
# print(listframes)
# frames
M0_frame, M1_frame, M2_frame = True, False, False 

# function to get distance b/w fingres (8 & 12)
def fingres_dist(hands, img):
    hand = hands[0]
    lmList = hand['lmList']
    bbox = hand['bbox']

    # calculate the distance between the two fingers using the bounding box size as a reference
    bbox_width, bbox_height = bbox[2], bbox[3]
    bbox_diagonal = sqrt(bbox_width ** 2 + bbox_height ** 2)
    length, info, img = hand_detector.findDistance(lmList[8][:2], lmList[12][:2], img)
    dist_len = (2.5 / bbox_diagonal) * length
    dist_len = int(dist_len * 100)
    return dist_len


# function for hand tracking
def hand_tracking(img, m_img):
    global M0_frame, M1_frame, M2_frame, imageNumber
    hands,img = hand_detector.findHands(img, flipType=False, draw=True)
    if hands:
        dist_len = fingres_dist(hands, img)  # get distance b/w fingres (8 & 12)
        
        # capture hands and track
        lmList = hands[0]['lmList']
        cv2.circle(m_img, lmList[8][:2], 12, (0, 0, 255), cv2.FILLED)
        
        # Page 1
        # select options
        if dist_len < 60 and M0_frame and lmList[8][0]>530 and lmList[8][1]>245 and \
                                lmList[8][0]<810 and lmList[8][1]<300:
            print('Measure body clicked')
            M0_frame, M1_frame, M2_frame = False, True, False 

        elif dist_len < 60 and M0_frame and lmList[8][0]>530 and lmList[8][1]>305 and \
                                lmList[8][0]<810 and lmList[8][1]<355:
            print('Try Shirt')
            M0_frame, M1_frame, M2_frame = False, False, True 
            # m_img = img

        elif dist_len < 60 and M0_frame and lmList[8][0]>530 and lmList[8][1]>360 and \
                                lmList[8][0]<810 and lmList[8][1]<410:
            print('Exit')
            # break

        # Page 1
        hand = hands[0]
        fingers = hand_detector.fingersUp(hand)  # List of which fingers are up
        # print(fingers)
        if fingers == [1, 0, 0, 0, 1]:
            print("Left")
            M0_frame, M1_frame, M2_frame = True, False, False 

        # page 2
        # select options
        if dist_len < 60 and M2_frame and lmList[8][0]>1100 and lmList[8][1]>345 and \
                                lmList[8][0]<1160 and lmList[8][1]<405 and imageNumber<len(listShirts)-1:
            print('Shirt change', os.path.join(shirtFolderPath, listShirts[imageNumber]))
            imageNumber +=1
            time.sleep(0.8)
        elif dist_len < 60 and M2_frame and lmList[8][0]>100 and lmList[8][1]>345 and \
                                lmList[8][0]<160 and lmList[8][1]<405 and imageNumber>0:
            print('Shirt change', os.path.join(shirtFolderPath, listShirts[imageNumber]))
            imageNumber -=1
            time.sleep(0.5)

    return img, m_img


def frame_2(img):
    global imageNumber
    # Get points from posenet model
    # img = pose_detector.findPose(img)
    img = pose_detector.findPose(img,draw=False)
    lmList, bboxInfo = pose_detector.findPosition(img, bboxWithHands=False, draw=False)
    
    try:
        if lmList:
            # center = bboxInfo["center"]
            lm11 = lmList[11][1:3]
            lm12 = lmList[12][1:3]

            # shirt read from folder
            imgShirt = cv2.imread(os.path.join(shirtFolderPath, listShirts[imageNumber]), cv2.IMREAD_UNCHANGED)
            # print(lm11)
            
            # shirt resize/scaling shirt 
            widthOfShirt = int((lm11[0] - lm12[0]) * fixedRatio)
            # multiply aspect ratio with width of shirt to get height
            imgShirt = cv2.resize(imgShirt, (widthOfShirt, int(widthOfShirt * aspect_ratio)))      
            
            # overlay shirt on body
            # print((lm11[0] - lm12[0]) / 190)
            currentScale = (lm11[0] - lm12[0]) / 190
            offset = int(44 * currentScale), int(48 * currentScale)
            
            # overlay on video stream
            try:
                img = cvzone.overlayPNG(img, imgShirt, (lm12[0] - offset[0], lm12[1] - offset[1]))
            except:
                pass        
    except:
        pass
    return img


def frame_1(img):
    img = pose_detector.findPose(img)
    lmList = pose_detector.findPosition(img,draw=False)
    if len(lmList[0])>0:
        # left hand
        dist_11_13, info=hand_detector.findDistance(lmList[0][13][1:-1],lmList[0][11][1:-1])
        dist_11_15, info=hand_detector.findDistance(lmList[0][13][1:-1],lmList[0][15][1:-1])
        # right hand
        dist_12_14, info=hand_detector.findDistance(lmList[0][12][1:-1],lmList[0][14][1:-1])
        dist_14_16, info=hand_detector.findDistance(lmList[0][14][1:-1],lmList[0][16][1:-1])
        # shoulder
        dist_11_12, info=hand_detector.findDistance(lmList[0][11][1:-1],lmList[0][12][1:-1])
        # hip
        dist_23_24, info=hand_detector.findDistance(lmList[0][23][1:-1],lmList[0][24][1:-1])
        # left leg
        dist_23_25, info=hand_detector.findDistance(lmList[0][23][1:-1],lmList[0][25][1:-1])
        dist_25_27, info=hand_detector.findDistance(lmList[0][25][1:-1],lmList[0][27][1:-1])
        # right leg
        dist_24_26, info=hand_detector.findDistance(lmList[0][24][1:-1],lmList[0][26][1:-1])
        dist_26_28, info=hand_detector.findDistance(lmList[0][26][1:-1],lmList[0][28][1:-1])

        # Calculate the scaling factor
        scaling_factor = body_height / dist_11_12

        # Calculate the inches per pixel value
        inches_per_pixel = scaling_factor * distance_from_camera / image_width_in_pixels
        data = []
        cv2.putText(img, f'Left Hand Size :{round((dist_11_15+dist_11_13) * inches_per_pixel,2)} inches', (10, 20), font, font_scale, font_color, thickness)
        data.append(round((dist_11_15+dist_11_13) * inches_per_pixel,2))
        cv2.putText(img, f'Right Hand Size :{round((dist_12_14+dist_14_16) * inches_per_pixel,2)} inches', (10, 50), font, font_scale, font_color, thickness)
        data.append(round((dist_12_14+dist_14_16) * inches_per_pixel,2))    
        cv2.putText(img, f'Shoulder Size:{round((dist_11_12) * inches_per_pixel,2)} inches', (10, 80), font, font_scale, font_color, thickness)
        data.append(round((dist_11_12) * inches_per_pixel,2))
        cv2.putText(img, f'Chest Size:{round((dist_11_12/2) * 2.7 * inches_per_pixel,2)} inches', (10, 110), font, font_scale, font_color, thickness)
        data.append(round((dist_11_12) * inches_per_pixel,2))
        cv2.putText(img, f'Waist Size:{round((dist_23_24) * 2.5 * inches_per_pixel,2)} inches', (10, 140), font, font_scale, font_color, thickness)
        data.append(round((dist_23_24) * inches_per_pixel,2))
        cv2.putText(img, f'Left Leg :{round((dist_23_25+dist_25_27) * inches_per_pixel,2)} inches', (10, 170), font, font_scale, font_color, thickness)
        data.append(round((dist_23_25+dist_25_27) * inches_per_pixel,2))
        cv2.putText(img, f'Right Leg :{round((dist_24_26+dist_26_28) * inches_per_pixel,2)} inches', (10, 200), font, font_scale, font_color, thickness)
        data.append(round((dist_24_26+dist_26_28) * inches_per_pixel,2))
        sock.sendto(str.encode(str(data)), serverAddressPort)

    return img



while True:
    success, img = cap.read()
    main_img = cv2.imread(os.path.join(frameFolderPath, f'{back_imageNumber}.png'), cv2.IMREAD_UNCHANGED)
    img = cv2.flip(img, 1)
    img, main_img = hand_tracking(img, main_img)
    
    # print(M1_frame, M2_frame)
    if M0_frame:
        back_imageNumber = 0    
    elif M1_frame:
        # back_imageNumber = 1
        body_out = cv2.imread(r"C:\Users\PC\Downloads\output-onlinepngtools1.png", cv2.IMREAD_UNCHANGED)
        main_img = cvzone.overlayPNG(img, body_out, [490, 20])
        h, w, _ = main_img.shape
        # Draw vertical line
        cv2.line(main_img, (w//2, 0), (w//2, h), (0, 0, 255), thickness=2)
        main_img = frame_1(main_img)

    elif M2_frame: 
        back_imageNumber = 2
        img = frame_2(img)
        imgShirt_show = cv2.imread(os.path.join(shirtFolderPath, listShirts[imageNumber]), cv2.IMREAD_UNCHANGED)
        btn_r = cv2.imread(os.path.join(buttonFolderPath, 'btn_r.png'), cv2.IMREAD_UNCHANGED)
        btn_l = cv2.imread(os.path.join(buttonFolderPath, 'btn_l.png'), cv2.IMREAD_UNCHANGED)
        # Remove alpha channel from mask image

        main_img = cvzone.overlayPNG(img, btn_r, (1100, 350))
        main_img = cvzone.overlayPNG(main_img, btn_l, (100, 350))


        # main_img = img

    cv2.imshow("Image", main_img)

    # to stop program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()