# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import sys

if len(sys.argv) != 2:
    print("You should specify a image folder where the images will be stored")
    
folder = sys.argv[1]
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (224, 224)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=camera.resolution)
# allow the camera to warmup
time.sleep(0.1)
ind = 0
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array
    # show the frame
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    # if the `q` key was pressed, break from the loop
    if key == ord("s"):
        # save the image
        image_name = folder + "/img" + str(ind) + ".jpg"
        cv2.imwrite(image_name, image)
        ind+=1
    if key == ord("q"): 
        break
cv2.destroyAllWindows()
