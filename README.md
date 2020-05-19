# Face Recognition using Transfer Learning on VGG16
## Transfer Learning
In the data science world, we face a lot of situation which require a lot of computational power or resources like RAM, CPU or GPU, etc. to train our models. To cope up with this situation one of the best ways is to use the pre-trained model as in Transfer Learning. This will also help to achieve more accuracy with less amount of data.

## VGG16
VGG16 is a convolutional neural network model proposed by K. Simonyan and A. Zisserman from the University of Oxford in the paper “Very Deep Convolutional Networks for Large-Scale Image Recognition”. The model achieves 92.7% top-5 test accuracy in ImageNet, which is a dataset of over 14 million images belonging to 1000 classes. It was one of the famous models submitted to ILSVRC-2014. It makes the improvement over AlexNet by replacing large kernel-sized filters (11 and 5 in the first and second convolutional layer, respectively) with multiple 3×3 kernel-sized filters one after another. VGG16 was trained for weeks and was using NVIDIA Titan Black GPUs.

## Prerequisites
Install following libraires:
```
conda install tensorflow keras opencv-python pillow numpy
```
## Creating images dataset using opencv throgh webcam
```
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "ename": "error",
     "evalue": "OpenCV(4.2.0) C:\\projects\\opencv-python\\opencv\\modules\\objdetect\\src\\cascadedetect.cpp:1689: error: (-215:Assertion failed) !empty() in function 'cv::CascadeClassifier::detectMultiScale'\n",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31merror\u001b[0m                                     Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-1-bbef18c16313>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m     30\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     31\u001b[0m     \u001b[0mret\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mframe\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mcap\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mread\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 32\u001b[1;33m     \u001b[1;32mif\u001b[0m \u001b[0mface_extractor\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mframe\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;32mis\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     33\u001b[0m         \u001b[0mcount\u001b[0m \u001b[1;33m+=\u001b[0m \u001b[1;36m1\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     34\u001b[0m         \u001b[0mface\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mcv2\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mresize\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mface_extractor\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mframe\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m(\u001b[0m\u001b[1;36m200\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;36m200\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m<ipython-input-1-bbef18c16313>\u001b[0m in \u001b[0;36mface_extractor\u001b[1;34m(img)\u001b[0m\n\u001b[0;32m     11\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     12\u001b[0m     \u001b[0mgray\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mcv2\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mcvtColor\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mimg\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mcv2\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mCOLOR_BGR2GRAY\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 13\u001b[1;33m     \u001b[0mfaces\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mface_classifier\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdetectMultiScale\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mgray\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;36m1.3\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;36m5\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     14\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     15\u001b[0m     \u001b[1;32mif\u001b[0m \u001b[0mfaces\u001b[0m \u001b[1;32mis\u001b[0m \u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31merror\u001b[0m: OpenCV(4.2.0) C:\\projects\\opencv-python\\opencv\\modules\\objdetect\\src\\cascadedetect.cpp:1689: error: (-215:Assertion failed) !empty() in function 'cv::CascadeClassifier::detectMultiScale'\n"
     ]
    }
   ],
   "source": [
    "import cv2\n",
    "import numpy as np\n",
    "\n",
    "# Load HAAR face classifier\n",
    "face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')\n",
    "\n",
    "# Load functions\n",
    "def face_extractor(img):\n",
    "    # Function detects faces and returns the cropped face\n",
    "    # If no face detected, it returns the input image\n",
    "    \n",
    "    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)\n",
    "    faces = face_classifier.detectMultiScale(gray, 1.3, 5)\n",
    "    \n",
    "    if faces is ():\n",
    "        return None\n",
    "    \n",
    "    # Crop all faces found\n",
    "    for (x,y,w,h) in faces:\n",
    "        cropped_face = img[y:y+h, x:x+w]\n",
    "\n",
    "    return cropped_face\n",
    "\n",
    "# Initialize Webcam\n",
    "cap = cv2.VideoCapture(0)\n",
    "count = 0\n",
    "\n",
    "# Collect 500 samples of your face from webcam input\n",
    "while True:\n",
    "\n",
    "    ret, frame = cap.read()\n",
    "    if face_extractor(frame) is not None:\n",
    "        count += 1\n",
    "        face = cv2.resize(face_extractor(frame), (200, 200))\n",
    "        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)\n",
    "\n",
    "        # Save file in specified directory with unique name\n",
    "        file_name_path = 'C://Users//vivekPC//Desktop//MLOps WS//FRTL//FRsamples//image' + str(count) + '.jpg'\n",
    "        cv2.imwrite(file_name_path, face)\n",
    "\n",
    "        # Put count on images and display live count\n",
    "        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)\n",
    "        cv2.imshow('Face Cropper', face)\n",
    "        \n",
    "    else:\n",
    "        print(\"Face not found\")\n",
    "        pass\n",
    "\n",
    "    if cv2.waitKey(1) == 13 or count == 500: #13 is the Enter Key\n",
    "        break\n",
    "        \n",
    "cap.release()\n",
    "cv2.destroyAllWindows()      \n",
    "print(\"Collecting Samples Complete\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
```
