# import the necessary packages
import numpy as np
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default=None, help='Path to input file')
args = parser.parse_args()

model_path = "../Models/OpenCV/opencv_face_detector_uint8.pb"
model_pbtxt = "../Models/OpenCV/opencv_face_detector.pbtxt"

# Se carga nuestro modelo serializado desde el disco
net = cv2.dnn.readNetFromTensorflow(model_path, model_pbtxt)

image = cv2.imread(args.input)
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
cv2.waitKey(0)