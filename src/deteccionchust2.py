import cv2
import argparse
import dlib
import os
import imutils
import copy
import numpy as np

dataPath = 'your_data_path'
faceClassif = cv2.CascadeClassifier('../Models/firstTests/haarcascade_frontalface_default.xml')
hogFaceDetector = dlib.get_frontal_face_detector()


# Function to detect faces in an image
def FaceDetection(image, method):
    x,y,w,h=0,0,0,0
    if (method == 1):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = faceClassif.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.imshow('image',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if (method == 2):
        faces = hogFaceDetector(image, 1)
        for (i, rect) in enumerate(faces):
            x = rect.left()
            y = rect.top()
            w = rect.right() - x
            h = rect.bottom() - y
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return x,y,w,h

# Function for including a person in the dataset from a video
# It is necessary to train the model each time someone is added.
def saveFacesFromVideo(personName, method):
    personPath = dataPath + '/' + personName
    if not os.path.exists(personPath):
        os.makedirs(personPath)

    # Capture from streaming
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    ## Capture from video
    # cap = cv2.VideoCapture('video_path')
    count = 0
    while True:
        ret, frame = cap.read()
        if ret == False: break
        frame = imutils.resize(frame, width=640)
        auxFrame = frame.copy()
        x,y,w,h = FaceDetection(auxFrame, method)
        if x==0 and y==0 and w==0 and h==0: continue
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face = auxFrame[y:y + h, x:x + w]
        face = cv2.resize(face, (150, 150), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(personPath + '/face_{}.jpg'.format(count), face)
        count = count + 1
        # cv2.imshow('frame', frame)
        k = cv2.waitKey(1)
        if k & 0xFF == ord("q") or count >= 300: break
    cap.release()
    cv2.destroyAllWindows()


# Function for including a person in the dataset from an image
# It is necessary to train the model each time someone is added.
def saveFacesFromImage(personName, method, image):
    personPath = dataPath + '/' + personName
    if not os.path.exists(personPath):
        os.makedirs(personPath)

    image = imutils.resize(image, width=640)
    auxFrame = image.copy()
    faces = FaceDetection(auxFrame, method)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(personPath + '/name.jpg', rostro)

# Function to train facial recognition model and save it in an xml file.
def train_model(method):
    peopleList = os.listdir(dataPath)
    labels = []
    facesData = []
    label = 0
    for nameDir in peopleList:
        personPath = dataPath + '/' + nameDir
        for fileName in os.listdir(personPath):
            labels.append(label)
            facesData.append(cv2.imread(personPath + '/' + fileName, 0))
        label = label + 1

    # Eigenfaces
    if method == 1:
        face_recognizer = cv2.face.EigenFaceRecognizer_create()
        face_recognizer.train(facesData, np.array(labels))
        face_recognizer.write('../Models/firstTests/eigenfaces_model.xml')

    # Fisherfaces
    if method == 2:
        face_recognizer = cv2.face.FisherFaceRecognizer_create()
        face_recognizer.train(facesData, np.array(labels))
        face_recognizer.write('../Models/firstTests/fisherfaces_model.xml')
    # LBP
    if method == 3:
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.train(facesData, np.array(labels))
        face_recognizer.write('../Models/firstTests/lbp_model.xml')


# Face recognition via streaming or video
def FaceRecognition(method, train=False):

    if train:
        train_model(method)

    imagePaths = os.listdir(dataPath)

    # Eigenfaces
    if method == 1:
        face_recognizer = cv2.face.EigenFaceRecognizer_create()
        face_recognizer.read('../Models/firstTests/eigenfaces_model.xml')
    # Fisherfaces
    if method == 2:
        face_recognizer = cv2.face.FisherFaceRecognizer_create()
        face_recognizer.read('../Models/firstTests/fisherfaces_model.xml')
    # LBP
    if method == 3:
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.read('../Models/firstTests/lbp_model.xml')

    # Capture from streaming
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    ## Capture from video
    # cap = cv2.VideoCapture('video_path')

    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    while True:
        ret, frame = cap.read()
        if ret == False: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = gray.copy()
        faces = faceClassif.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face = auxFrame[y:y + h, x:x + w]
            face = cv2.resize(face, (150, 150), interpolation=cv2.INTER_CUBIC)
            result = face_recognizer.predict(face)
            cv2.putText(frame, '{}'.format(result), (x, y - 5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

            if method == 1 and result[1] < 4500 or method == 2 and result[1] < 750 or method == 3 and result[1] < 70:
                cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x, y - 25), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'Unknown', (x, y - 25), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # cv2.imshow('frame', frame)
        k = cv2.waitKey(1)
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None, help='Path to input file (only for images)')
    parser.add_argument('--detection', type=int, default=1, help='1 for Viola-Jones and 2 for HOG')
    parser.add_argument('--recognition', type=int, default=1, help='1 for Eigenfaces, 2 for Fisherfaces and 3 for LBP')
    parser.add_argument("--train-model", action="store_true", help="True for train model")


    args = parser.parse_args()

    # if (args.input):
    #     image = cv2.imread(args.input)
    # FaceDetection(image, args.detection)


    saveFacesFromVideo('yourname', args.detection)

    FaceRecognition(args.recognition, args.train_model)
