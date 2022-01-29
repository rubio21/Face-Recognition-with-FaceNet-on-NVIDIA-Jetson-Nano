import time
import argparse
import cv2
import imutils
from imutils.video import VideoStream
import copy
import numpy as np



model_path = "../Models/OpenCV/opencv_face_detector_uint8.pb"
model_pbtxt = "../Models/OpenCV/opencv_face_detector.pbtxt"
dataset_path="../faces/"

# Se carga nuestro modelo serializado desde el disco
net = cv2.dnn.readNetFromTensorflow(model_path, model_pbtxt)

def DetectFaces(image):
    # image = cv2.imread(args.input)
    height, width, channels = image.shape
    # Se carga la imagen de entrada y se construye un blob de entrada para la imagen,
    # cambiando el tamaño a un valor fijo de 300x300 píxeles y normalizándolo
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
    # Se pasa la blob por la red y se obtienen las detecciones y predicciones
    net.setInput(blob)
    detections = net.forward()

    faces = []

    # Bucle sobre las detecciones
    for i in range(detections.shape[2]):
        # Se extrae la confianza (probabilidad) asociada a la predicción
        confidence = detections[0, 0, i, 2]
        # Se filtran las detecciones débiles, comprobando que la confianza es mayor que el threshold
        if confidence > 0.5:
            # Se calculan las coordenadas que delimitan el rectángulo del rostro
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
    if(video_path==None):
        cap = cv2.VideoCapture(0)
        image_flip=True
    else:
        cap = cv2.VideoCapture(video_path)
    time.sleep(2.0)

    if (train):
        confidenceMax = 0.49
        bestFrame=np.zeros([500,500,3])

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
    if train and confidenceMax > 0.49:
        cv2.imwrite(dataset_path + 'your_name.jpg', bestFrame)

    cv2.destroyAllWindows()
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None, help='Path to input file')
    parser.add_argument("--train", action="store_true", help="Captures the best face of the video and stores it in the dataset")
    args = parser.parse_args()

    DetectFacesVideo(args.input, args.train)
