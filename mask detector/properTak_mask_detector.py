from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2
import os
import time

detections = None

# Function to detect face then in Region of Interest (ROI) detect mask-wearing condition
def detect_and_predict_mask(frame, faceNet, maskNet, threshold):
    global detections
    # Get height,width
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faces = []
    locs = []  # locations
    preds = [] # predictions
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            locs.append((startX, startY, endX, endY))
            preds.append(maskNet.predict(face)[0].tolist())
    return (locs, preds)


# Settings and Configurations for detecting mask-wearing condition
MASK_MODEL_PATH = os.path.join(os.getcwd(),"model.h5")
FACE_MODEL_PATH = os.path.join(os.getcwd(),"face_detector")
THRESHOLD = 0.5

print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([FACE_MODEL_PATH, "deploy.prototxt"])
weightsPath = os.path.sep.join([FACE_MODEL_PATH, "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

print("[INFO] loading face mask detector model...")
maskNet = load_model(MASK_MODEL_PATH)

# Real-time detection on mask-wearing conditions via webcam feed
def live_cam():
    print("[INFO] starting video stream...")
    vs = VideoStream(0).start()
    time.sleep(2.0)

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        original_frame = frame.copy()
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet, THRESHOLD)
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (correct, incorrect, without) = pred
            label = "Mask" if correct > incorrect else "No Mask"
            # Case for mask-wearing correctly
            wi = incorrect + without
            # Case for mask-wearing not correctly
            cw = correct + without
            # Case for there's no mask-wearing
            ci = correct + incorrect
            # OpenCV color is in (B,G,R)
            if correct > wi:
                label = "Mask"
                color = (0, 255, 0) # Green
            elif incorrect > cw:
                label = "Incorrect"
                color = (3, 148, 251)  # Orange
            elif without > ci:
                label = "No Mask"
                color = (0, 0, 255) # Red
            else:
                label = "----"
            white_color = (255, 255, 255)
            # Showing percentage
            label = "{}{:.2f}%".format(label, max(correct, incorrect, without) * 100)
            dy = startY - 15
            for i, line in enumerate(label.split('\n')):
                cv2.putText(original_frame, line, (startX, dy), cv2.cv2.FONT_HERSHEY_DUPLEX, 0.3, white_color)
                dy = dy + 10
            cv2.rectangle(original_frame, (startX, startY), (endX, endY), color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, startY - 23), color, -2)

        cv2.addWeighted(frame, 0.5, original_frame, 0.5, 0, frame)
        frame = cv2.resize(frame, (860, 490))
        # Display the output
        cv2.imshow("properTak? Mask Detector", frame)
        # To quit, press q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    vs.stop()

# Real-time detection on mask-wearing conditions on pictures
def detect_pic(pic_path):
    frame = cv2.imread(pic_path)
    frame = imutils.resize(frame, width=400)
    original_frame = frame.copy()
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet, THRESHOLD)
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (correct, incorrect, without) = pred
        label = "Mask" if correct > incorrect else "No Mask"
        # Case for mask-wearing correctly
        wi = incorrect + without
        # Case for mask-wearing not correctly
        cw = correct + without
        # Case for there's no mask-wearing
        ci = correct + incorrect
        # OpenCV color is in (B,G,R)
        if correct > wi:
            label = "Mask"
            color = (0, 255, 0) # Green
        elif incorrect > cw:
            label = "Incorrect"
            color = (3, 148, 251)  # Orange
        elif without > ci:
            label = "No Mask"
            color = (0, 0, 255) # Red
        else:
            label = "----"
        white_color = (255, 255, 255)
        # Shwoing percentage
        label = "{}{:.2f}%".format(label, max(correct, incorrect, without) * 100)
        dy = startY - 15
        for i, line in enumerate(label.split('\n')):
            cv2.putText(original_frame, line, (startX, dy), cv2.cv2.FONT_HERSHEY_DUPLEX, 0.3, white_color)
            dy = dy + 10
        cv2.rectangle(original_frame, (startX, startY), (endX, endY), color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, startY - 23), color, -2)

    cv2.addWeighted(frame, 0.5, original_frame, 0.5, 0, frame)
    frame = cv2.resize(frame, (860, 490))
    ## Display the output
    ## Make windows resizeable
    # cv2.namedWindow('properTak? Mask Detector', cv2.WINDOW_NORMAL)
    # cv2.imshow('properTak? Mask Detector', frame)
    # cv2.waitKey(0)
    return frame

# Original Passport Photo: "C:/Users/Jasmoon/Pictures/face_to_be_mask/chan jia liang.jpg"
# Path to input files: "C:/Users/Jasmoon/PycharmProjects/SoftComp/generated"

## A for-loop to get result on input files
# parent_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
# input_path = os.path.join(parent_path,"mask-to-face","generated")
# for root, dirs, files in os.walk(input_path):
#     for filename in files:
#         res=detect_pic(os.path.join(input_path,filename))
#         # Retrieve back original size
#         res = cv2.resize(res,(1121,1642))
#         saved_name = os.path.join(os.getcwd(),"output",filename)
#         cv2.imwrite(saved_name,res)

## Individual test case for original passport photo
# res= detect_pic("C:/Users/Jasmoon/Pictures/face_to_be_mask/chan jia liang.jpg")
## Retrieve back original size
# res = cv2.resize(res,(1121,1642))
# saved_name = os.path.join(os.getcwd(),"output","no mask.png")
# cv2.imwrite(saved_name,res)

# To start webcam capture for detecting mask-wearing conditions
live_cam()


