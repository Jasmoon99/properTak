# properTak

## Project Structure

The `mask detector` folder included files needed to train the model and as well to detect mask wearing conditions on either still pictures or live webcam capture

***
- **`dataset`** folder included `correct`, `incorrect`, `without` folder  which contains sample images used to train for the model classifier 
- **`face_detector`** folder included DNN model for face detector and its deploy.prototxt
  - It is used to detect face and draw the region of interest (ROI) in the process of detecting the mask-wearing condition
- **`output`** folder contains images and and animated GIF result to show the mask wearing conditions on both still picture and on webcam respectively
- `model.h5` is the outcome from `train_mask_detection_model.py` -- our script used to train the model for mask-wearing condition
   - It is loaded and used in `properTak_mask_detector.py` -- our script file to detect mask-wearing conditions on both picture and live webcam feed 
***
- **`mask-to-face`** folder included files needed to generate different scenarios of mask-wearing pictures such as:
  - **Correct Situation** ✅
    - Nose, Mouth, Chin are all fully covered
  - **Incorrect Situation** ❎
    - Nose not fully covered
    - Nose not covered at all
    - Only mouth covered
    - Below chin
- **`generated`** folder contains sample mask-wearing images generated using mask-to-face mapping technique
- `shape_predictor_68_face_landmarks.dat` is the module by **dlib** which will return 68 face landmarks on detected human faces
  - It is really useful in providing a boundary that helps to map the face mask on human faces.
- `apply_mask_to_face.py` is the Python script used to simulate different mask wearing conditions with thw help of **dlib** and **OpenCV**
***

```
properTak/
├── mask detector/
│   ├── dataset/
│   │   ├── correct/
│   │   │   ├── 00000_Mask.jpg
│   │   │   ├── 00001_Mask.jpg
│   │   │   ├── 00002_Mask.jpg
│   │   │   ├── 00003_Mask.jpg
│   │   │   └── 00004_Mask.jpg
│   │   ├── incorrect/
│   │   │   ├── 00000_Mask_Mouth_Chin.jpg
│   │   │   ├── 00001_Mask_Chin.jpg
│   │   │   ├── 00001_Mask_Mouth_Chin.jpg
│   │   │   ├── 00002_Mask_Mouth_Chin.jpg
│   │   │   └── 00003_Mask_Mouth_Chin.jpg
│   │   └── without/
│   │       ├── 00000.png
│   │       ├── 00001.png
│   │       ├── 00002.png
│   │       ├── 00003.png
│   │       └── 00004.png
│   ├── face_detector/
│   │   ├── deploy.prototxt
│   │   └── res10_300x300_ssd_iter_140000.caffemodel
│   ├── output/
│   │   ├── no mask.png
│   │   ├── surgical0_below chin.png
│   │   ├── surgical0_fully_covered.png
│   │   ├── surgical0_nose_not_cover.png
│   │   ├── surgical0_nose_not_cover_chin.png
│   │   ├── surgical0_nose_not_fully_cover.png
│   │   ├── surgical1_below chin.png
│   │   ├── surgical1_fully_covered.png
│   │   ├── surgical1_nose_not_cover.png
│   │   ├── surgical1_nose_not_cover_chin.png
│   │   ├── surgical1_nose_not_fully_cover.png
│   │   └── Result on live webcam feed.gif
│   ├── model.h5
│   ├── plot.png
│   ├── properTak_mask_detector.py
│   └── train_mask_detection_model.py
└── mask-to-face/ 
    ├── generated/
    │   ├── surgical0_below chin.png
    │   ├── surgical0_fully_covered.png
    │   ├── surgical0_nose_not_cover.png
    │   ├── surgical0_nose_not_cover_chin.png
    │   ├── surgical0_nose_not_fully_cover.png
    │   ├── surgical1_below chin.png
    │   ├── surgical1_fully_covered.png
    │   ├── surgical1_nose_not_cover.png
    │   ├── surgical1_nose_not_cover_chin.png
    │   └── surgical1_nose_not_fully_cover.png
    ├── apply_mask_to_face.py
    ├── cloth-mask.png
    ├── cloth-mask2.png
    ├── cloth-mask3.png
    ├── n95-mask.png
    ├── shape_predictor_68_face_landmarks.dat
    ├── surgical-blue-face-mask.png
    └── surgical-blue-face-mask2.png

```
## Resources

### Training Dataset
1. [MaskedFace-Net](https://github.com/cabani/MaskedFace-Net)
    > `correct` folder images provided by Correctly Masked Face Dataset (CMFD) of MaskedFace-Net

    > `incorrect` folder images provided by Incorrectly Masked Face Dataset (IMFD) of MaskedFace-Net

2. [Flickr-Faces-HQ Dataset (FFHQ)](https://github.com/NVlabs/ffhq-dataset)
    > `without` folder images provider

### Dlib 68 Face Landmarks
[shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

### OpenCV DNN Face Detector
1. [Deploy.prototxt](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt)
2. [Model](https://github.com/opencv/opencv_3rdparty/blob/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel) 

## Environment
I have used anaconda to manage different packages and their respective versions which are compatible.
It really helps in setting up the environment for TensorFlow to run with GPU and as well the dlib library which conda can automatically resolve the CMake error

### To setup the environment,
1. Head to the [Anaconda](https://www.anaconda.com/products/individual) website and downlad the installer package
2. Follow the instructions to install the Anaconda
3. Then, create an environment of TensorFlow with GPU following this [tutorial](https://docs.anaconda.com/anaconda/user-guide/tasks/tensorflow/)
  > **Some reference**: [Conda command cheatsheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf#page=1&zoom=auto,-373,618)

### How to download packages that are not available by default (e.g. dlib) 
1. Search packages directly on [ANACONDA.ORG](https://anaconda.org/) | [CONDA-FORGE](https://conda-forge.org/feedstock-outputs/)
2. Then, click on the packages to get the conda command
3. To install, simply copy the command and run it in Anaconda Prompt, *provided you're in the environment you have created for this project*.

## How to configure the Conda interpreter in PyCharm
[Tutorial](https://www.jetbrains.com/help/pycharm/conda-support-creating-conda-virtual-environment.html)

## Output

![Demo](https://github.com/Jasmoon99/properTak/blob/main/mask%20detector/output/Result%20on%20live%20webcam%20feed.gif)
