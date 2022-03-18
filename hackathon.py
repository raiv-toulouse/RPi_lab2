# La Croix Flavor Detector - Machine Learning on Raspberry Pi
#
# Michael D'Argenio
# mjdargen@gmail.com
# https://dargenio.dev
# https://github.com/mjdargen
# Created: February 6, 2020
# Last Modified: February 27, 2020
#
# This program uses Tensorflow and OpenCV to detect objects in the video
# captured from your webcam. This program is meant to be used with machine
# learning models generated with Teachable Machine.
#
# Teachable Machine is a great machine learning model trainer and generator
# created by Google. You can use Teachable Machine to create models to detect
# objects in images, sounds in audio, or poses in images. For more info, go to:
# https://teachablemachine.withgoogle.com/
#
# For this project, you will be generating a image object detection model. Go
# to the website, click "Get Started" then go to "Image Project". Follow the
# steps to create a model. Export the model as a "Tensorflow->Keras" model.
#
# To run this code in your environment, you will need to:
#   * Install Python 3 packages & library dependencies
#       * Use installation shell script
#   * Export your teachable machine tensorflow keras model and unzip it.
#       * You need both the .h5 file and labels.txt
#   * Update model_path to point to location of your keras model
#   * Update labels_path to point to location of your labels.txt
#   * Adjust width and height of your webcam for your system
#       * Adjust frameWidth with your video feed width in pixels
#       * Adjust frameHeight with your video feed height in pixels
#       * My RPi Camera v1.3 works well with 1024x768
#   * Set your confidence threshold
#       * conf_threshold by default is 90
#   * If video does not show up properly, use the matplotlib implementation
#       * Uncomment "import matplotlib...."
#       * Comment out "cv2.imshow" and "cv2.waitKey" lines
#       * Uncomment plt lines of code below
#   * Run "sudo python3 hackathon.py"

import multiprocessing
import numpy as np
import cv2
import tensorflow.keras as tf
import pyttsx3
import math
import time
import RPi.GPIO as GPIO
from picamera.array import PiRGBArray
from picamera import PiCamera
# use matplotlib if cv2.imshow() doesn't work
# import matplotlib.pyplot as plt


# disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Pin Definitions
class_pin = [12,16,18,22]  # BCM pin 18, BOARD pin 12

# Pin Setup:
GPIO.setmode(GPIO.BOARD)  # BOARD pin-numbering scheme from Raspberry Pi
# set pin as an output pin with optional initial state of LOW
for i in range(len(class_pin)):
    GPIO.setup(class_pin[i], GPIO.OUT, initial=GPIO.LOW)

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (224, 224)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=camera.resolution)
# allow the camera to warmup
time.sleep(0.1)

# read labels.txt file to get labels
labels_path = "../labels.txt"
# open input file label.txt
labelsfile = open(labels_path, 'r')

# initialize classes and read in lines until there are no more
classes = []
line = labelsfile.readline()
while line:
    # retrieve just class name and append to classes
    classes.append(line.split(' ', 1)[1].rstrip())
    line = labelsfile.readline()
# close label file
labelsfile.close()

# load the teachable machine model previously trained by teachable_machine 
model_path = '../keras_model.h5'
model = tf.models.load_model(model_path, compile=False)

frameHeight = frameWidth = 224

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array

    # Create the array of the right shape to feed into the keras model.
    # We are inputting 1x 224x224 pixel RGB image.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # turn the image into a numpy array
    image_array = np.asarray(image)
    # normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # load the image into the array
    data[0] = normalized_image_array

    # run the prediction. In the prediction, we have the % for all classes for classification
    predictions = model.predict(data)

    # confidence threshold is 90% so a class is recognize if its confidence % is over this threshold
    conf_threshold = 90
    confidence = []
    conf_label = ""
    threshold_class = ""
    # create black border at bottom for labels
    per_line = 1  # number of classes per line of text
    bordered_frame = cv2.copyMakeBorder(
        image,
        top=0,
        bottom=30 + 15*math.ceil(len(classes)/per_line),
        left=0,
        right=0,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )
    # for each one of the classes
    for i in range(0, len(classes)):
        # scale prediction confidence to % and apppend to 1-D list
        confidence.append(int(predictions[0][i]*100))
        # put text per line based on number of classes per line
        if (i != 0 and not i % per_line):
            cv2.putText(
                img=bordered_frame,
                text=conf_label,
                org=(int(0), int(frameHeight+25+15*math.ceil(i/per_line))),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(255, 255, 255)
            )
            conf_label = ""
        # append classes and confidences to text for label
        conf_label += classes[i] + ": " + str(confidence[i]) + "%; "
        # prints last line
        if (i == (len(classes)-1)):
            cv2.putText(
                img=bordered_frame,
                text=conf_label,
                org=(int(0), int(frameHeight+25+15*math.ceil((i+1)/per_line))),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(255, 255, 255)
            )
            conf_label = ""
        # if above confidence threshold, send to queue
        if confidence[i] > conf_threshold:
            threshold_class = classes[i]
            GPIO.output(class_pin[i], True)  # switch on the LED
        else:
            GPIO.output(class_pin[i], False) # Switch off the LED
    # add label class above confidence threshold
    cv2.putText(
        img=bordered_frame,
        text=threshold_class,
        org=(int(20), int(frameHeight)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.75,
        color=(255, 255, 255)
    )
    
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    
    # original video feed implementation
    cv2.imshow("Capturing", bordered_frame)
    cv2.waitKey(10)

    # # if the above implementation doesn't work properly
    # # comment out two lines above and use the lines below
    # # will also need to import matplotlib at the top
    # plt_frame = cv2.cvtColor(bordered_frame, cv2.COLOR_BGR2RGB)
    # plt.imshow(plt_frame)
    # plt.draw()
    # plt.pause(.001)


