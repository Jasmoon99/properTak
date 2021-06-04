import time

import dlib
import cv2
import numpy as np
import os

# https://stackoverflow.com/questions/40895785/using-opencv-to-overlay-transparent-image-onto-another-image
# Function to add resized, transparent mask on the passport photo
def overlay_transparent(background, overlay, x, y):
    # Get passport photo width,height
    background_width = background.shape[1]
    background_height = background.shape[0]

    # If overlay is bigger, ignore and return
    if x >= background_width or y >= background_height:
        return background

    # Get resized, transparent mask width,height
    h, w = overlay.shape[0], overlay.shape[1]

    # Locate x,y as the top-left point and overlay them
    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    # IF there's no alpha channel, the fourth channel after RGB, then add one for that mask.png
    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0
    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background

# Form boundary area based on keypoint
def get_border(shape,keypoint):  # (68,2)
    res = []
    for idx,pt in enumerate(shape.parts()):
        if idx in keypoint:
            res.append([pt.x,pt.y])
    return res[0][0],res[3][1],res[2][0],res[1][1] # return the respective x1,y1,x2,y2 (the top-left and the bottom-right coordinates...)

# Function to peform mask-to-face mapping...
def wear_mask(img,mask,keypoint):
    # Face detection using Dlib Histogram of Oriented Gradients (HOG) Face Detector
    detector = dlib.get_frontal_face_detector()

    # Load the predictor that feedback 68 landmarks
    # Further details: https://www.researchgate.net/figure/The-68-specific-human-face-landmarks_fig4_331769278
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # Preprocess the passport photo before performing mask-to-face mapping..
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Ask the detector to find the bounding boxes of each face.
    # The second argument indicates that we should upsample the image x time.
    # This will make everything bigger and allow us to detect more faces if there's need (whereby faces in photo is really small).
    dets = detector(gray,0)

    # For every possible result of faces:
    for d in dets:
        # Use predictor to gain the face landmarks
        shape = predictor(gray, d)
        # Based on the keypoint, get the boundary for the mask to be apply on faces..
        x1,y1,x2,y2 = get_border(shape,keypoint)
        # resize accordingly
        mask_use = cv2.resize(mask, (x2-x1, y2-y1))
        result = overlay_transparent(img, mask_use, x1, y1)
        # Display the result of adding mask
        # cv2.namedWindow('Modified Passport Photo (Wearing Mask)', cv2.WINDOW_NORMAL)
        # cv2.imshow('Modified Passport Photo (Wearing Mask)', added_image)
        # cv2.waitKey(0)
    return result


if __name__ == "__main__":

    # Different set of kepoint to generate different wearing mask scenarios
    keypoint1 = [2, 8, 14, 28] # basically cover up the whole face until nose
    keypoint2 = [3,8,13,57] # below chin
    keypoint3 = [2,8,14,30] # nose_not_fully_cover
    keypoint4 = [2,8,14,33] # nose_not_cover
    keypoint5 = [2,8,14,51] # nose_not_cover_chin

    # Get filename to be saved for test data nanti
    def get_saved_name(mask_type,correctness):
        if mask_type == 0 or mask_type==1 or mask_type==2: # cloth
            saved_name=f"cloth{mask_type}_"
        elif mask_type == 3: # n95
            saved_name = "n95_"
        elif mask_type == 4 or mask_type == 5: # surgical1
            saved_name = f"surgical{mask_type-4}_"

        if correctness == 0:
            saved_name+="fully_covered"
        elif correctness == 1:
            saved_name += "below chin"
        elif correctness == 2:
            saved_name += "nose_not_fully_cover"
        elif correctness == 3:
            saved_name += "nose_not_cover"
        elif correctness == 4:
            saved_name += "nose_not_cover_chin"

        #extension for the image
        saved_name+=".png"
        return saved_name

    # "C:/Users/Jasmoon/Pictures/face_to_be_mask/chan jia liang.jpg"
    base_img = r"C:/Users/Jasmoon/Pictures/face_to_be_mask/chan jia liang.jpg"
    img = cv2.imread(base_img)

    # TODO: try with different type of mask and compare the result.
    mask=['cloth-mask.png','cloth-mask2.png','cloth-mask3.png','n95-mask.png','surgical-blue-face-mask.png','surgical-blue-face-mask2.png']
    # Declare a list to store different keypoints
    keypoints=[]
    keypoints.append(keypoint1)
    keypoints.append(keypoint2)
    keypoints.append(keypoint3)
    keypoints.append(keypoint4)
    keypoints.append(keypoint5)

    # # generate all kind of masks on face with different positions
    # for mask_type, x in enumerate(mask):
    #     # With flag unchanged for reading in alpha channel as well (if applicable)
    #     mask_img = cv2.imread(x, cv2.IMREAD_UNCHANGED)
    #     for correctness, y in enumerate(keypoints):
    #         ans = wear_mask(img,mask_img,y)
    #         saved_name= os.path.join(os.getcwd(),"generated",get_saved_name(mask_type,correctness))
    #         cv2.imwrite(saved_name, ans)

    ## Test case for individual case ##
    #mask 0-5 keypoint1-keypoint5
    mask = cv2.imread(mask[0],-1)
    ans= wear_mask(img,mask,keypoint1)
    # Display - Make windows resizeable
    # cv2.namedWindow('Modified Passport Photo (Wearing Mask)', cv2.WINDOW_NORMAL)
    # cv2.imshow('Modified Passport Photo (Wearing Mask)',ans)
    # cv2.waitKey(0)
    saved_name= os.path.join(os.getcwd(),"generated",get_saved_name(0,0))
    cv2.imwrite(saved_name,ans)



