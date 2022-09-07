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
#       * Adjust FRAME_HEIGHT with your video feed height in pixels
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
import zipfile
import math
import os
import serial

# use matplotlib if cv2.imshow() doesn't work
# import matplotlib.pyplot as plt

# 
# Constantes
#
CONF_THRESHOLD = 80 # confidence threshold.
CAMERA_INDEX = 0 # 0 pour la webcam intégrée au PV
# width & height of webcam video in pixels -> adjust to your size
# adjust values if you see black bars on the sides of capture window
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
PER_LINE = 2  # number of classes per line of text
COMMANDES = {'gauche':('5','2'), 'droite':('2','5'), 'avance':('5','5'), 'recule':('2','2'),
             'avance_rapide':('7','7'), 'stop':('x','x'), 'rien':('x','x')}
MODEL_ZIP_FILE = "/home/phil/Downloads/converted_keras.zip"
DESTINATION_FOLDER = "/home/phil/Icam/O3_lab2_IA/RPi_lab2"
LABEL_FILE = DESTINATION_FOLDER + "/labels.txt"
KERAS_MODEL_FILE = DESTINATION_FOLDER + "/keras_model.h5"

FROM_CAMERA = True # False si l'image vient d'un fichier

def init():
    arduino = serial.Serial('/dev/ttyACM0', baudrate=9600, timeout=2)
    # Extract Zip file
    if os.path.exists(MODEL_ZIP_FILE):
        if os.path.exists(LABEL_FILE):
            os.remove(LABEL_FILE)
        if os.path.exists(KERAS_MODEL_FILE):
            os.remove(KERAS_MODEL_FILE)
        with zipfile.ZipFile(MODEL_ZIP_FILE, mode="r") as archive:
            archive.extractall(DESTINATION_FOLDER)
        os.remove(MODEL_ZIP_FILE)
    # read .txt file to get labels
    labels_path = "labels.txt"
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

    # load the teachable machine model
    model_path = 'keras_model.h5'
    model = tf.models.load_model(model_path, compile=False)

    # initialize webcam video object
    cap = cv2.VideoCapture(CAMERA_INDEX)

    # set width and height in pixels
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    # enable auto gain
    cap.set(cv2.CAP_PROP_GAIN, 0)

    return cap, model, classes, arduino

def capture_image_for_tensorflow(cap):
    # Create the array of the right shape to feed into the keras model.
    # We are inputting 1x 224x224 pixel RGB image.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    if FROM_CAMERA:
        # capture image
        check, frame = cap.read()
        # mirror image - mirrored by default in Teachable Machine
        # depending upon your computer/webcam, you may have to flip the video
        frame = cv2.flip(frame, 1)

        # crop to square for use with TM model
        margin = int(((FRAME_WIDTH - FRAME_HEIGHT) / 2))
        square_frame = frame[0:FRAME_HEIGHT, margin:margin + FRAME_HEIGHT]

        # resize to 224x224 for use with TM model
        resized_img = cv2.resize(square_frame, (224, 224))
        # convert image color to go to model
        model_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        #model_img = resized_img
    else:  # from file
        image_filename = input("chemin du fichier image : ")
        square_frame = cv2.imread(image_filename)  # img is a 224*224 image
        model_img = square_frame

    # create black border at bottom for labels
    bordered_frame = cv2.copyMakeBorder(
        square_frame,
        top=0,
        bottom=30 + 15 * math.ceil(len(classes) / PER_LINE),
        left=0,
        right=0,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )

    # turn the image into a numpy array
    image_array = np.asarray(model_img)
    # normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # load the image into the array
    data[0] = normalized_image_array

    return data, bordered_frame

def ajouter_info_sur_image(conf_label, index, la_classe, la_proba):
    # put text per line based on number of classes per line
    if (index != 0 and not index % PER_LINE):
        cv2.putText(
            img=bordered_frame,
            text=conf_label,
            org=(int(0), int(FRAME_HEIGHT + 25 + 15 * math.ceil(index / PER_LINE))),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(255, 255, 255)
        )
        conf_label = ""
    # append classes and confidences to text for label
    conf_label += la_classe + ": " + str(la_proba) + "%; "
    # prints last line
    if (index == (len(classes) - 1)):
        cv2.putText(
            img=bordered_frame,
            text=conf_label,
            org=(int(0), int(FRAME_HEIGHT + 25 + 15 * math.ceil((index + 1) / PER_LINE))),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(255, 255, 255)
        )
        conf_label = ""

    return conf_label

def envoyer_commande_au_robot(class_command, arduino):  # avec class_command = 'left', 'right', ...
    (vitesse_moteur_gauche, vitesse_moteur_droit) = COMMANDES[class_command]  # cmd = (vitesse_moteur_gauche, vitesse_moteur_droit)
    ch_for_bluetooth = 's'+ vitesse_moteur_gauche + vitesse_moteur_droit
    # Envoyer ch_for_bluetooth par Bluetooth
    print(ch_for_bluetooth)
    arduino.write(ch_for_bluetooth.encode('utf-8'))  # envoi du message série

#
# main line code
# if statement to circumvent issue in windows
if __name__ == '__main__':
    # disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    cap, model, classes, arduino = init()

    # keeps program running forever until ctrl+c or window is closed
    while True:
        data, bordered_frame = capture_image_for_tensorflow(cap)

        # run the prediction
        predictions = model.predict(data)

        confidence = []
        conf_label = ""
        threshold_class = ""
        # for each one of the classes
        for i in range(0, len(classes)):
            # scale prediction confidence to % and apppend to 1-D list
            confidence.append(int(predictions[0][i]*100))

            # add proba on image for this class
            conf_label = ajouter_info_sur_image(conf_label, i, classes[i], confidence[i])

        # What is the best prediction?
        confidence_np = np.asarray(confidence)
        ind_best_predict = confidence_np.argmax()
        class_command = classes[ind_best_predict]
        if  confidence_np[ind_best_predict] < CONF_THRESHOLD:
            class_command = 'rien'

        # add label class above confidence threshold
        cv2.putText(
            img=bordered_frame,
            text=class_command,
            org=(int(20), int(FRAME_HEIGHT)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.75,
            color=(255, 255, 255)
        )

        # original video feed implementation
        print(class_command)
        cv2.imshow("Capturing", bordered_frame)
        cv2.waitKey(10)

        envoyer_commande_au_robot(class_command, arduino)



