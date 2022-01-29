import time
import argparse
import cv2
from imutils.video import VideoStream
import copy
import numpy as np



model_path = "../Models/OpenCV/opencv_face_detector_uint8.pb"
model_pbtxt = "../Models/OpenCV/opencv_face_detector.pbtxt"
dataset_path="../faces/"

# Our serialised model is loaded from disk
net = cv2.dnn.readNetFromTensorflow(model_path, model_pbtxt)

def DetectFaces(image):
    height, width, channels = image.shape
    # The input image is loaded and an input blob is constructed for the image,
    # resized to a fixed value of 300x300 pixels and normalised.
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
    # The blob is passed through the network and the detections and predictions are obtained.
    net.setInput(blob)
    detections = net.forward()

    faces = []

    # Loop over detections
    for i in range(detections.shape[2]):
        # The confidence associated with the prediction is extracted.
        confidence = detections[0, 0, i, 2]
        # Weak detections are filtered out, checking that the confidence is higher than the threshold.
        if confidence > 0.5:
            # The coordinates delimiting the rectangle of the face are calculated.
            text = "{:.2f}%".format(confidence * 100)
            x1 = int(detections[0, 0, i, 3] * width)
            y1 = int(detections[0, 0, i, 4] * height)
            x2 = int(detections[0, 0, i, 5] * width)
            y2 = int(detections[0, 0, i, 6] * height)
            faces.append([x1, y1, x2 - x1, y2 - y1])
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.putText(image, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    cv2.imshow("Output", image)
    return detections[0, 0, 0, 2]

def DetectFacesVideo(video_path, train):
    image_flip=False
    # Streaming
    if(video_path==None):
        cap = cv2.VideoCapture(0)
        image_flip=True
    else:
        # Video
        cap = cv2.VideoCapture(video_path)
    time.sleep(2.0)

    if (train):
        confidenceMax = 0.49
        bestFrame=np.zeros([500,500,3])

    # Looping during video or streaming
    while True:
        ret, image = cap.read()
        if image_flip: image=cv2.flip(image, 1)
        if not ret: return
        image = imutils.resize(image, width=400)
        confidence = DetectFaces(copy.deepcopy(image))

        if (train and confidence>confidenceMax):
            bestFrame=copy.deepcopy(image)
            confidenceMax=copy.deepcopy(confidence)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    if (video_path==0):
        vs.stop()
    # The best face is saved if the train parameter indicates so
    if train and confidenceMax > 0.49:
        cv2.imwrite(dataset_path + 'your_name.jpg', bestFrame)

    cv2.destroyAllWindows()
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None, help='Path to input file')
    parser.add_argument("--train", action="store_true", help="Captures the best face of the video and stores it in the dataset")
    args = parser.parse_args()
    
    # DetectFaces(args.input)
    DetectFacesVideo(args.input, args.train)
