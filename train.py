import cv2
import os
import numpy as np
import time

peopleList = os.listdir('Data/')
print('Lista de personas: ', peopleList)
labels = []
facesData = []
label = 0

tiempoInicio=time.time()
for nameDir in peopleList:
    personPath = 'Data/' + nameDir
    print('Leyendo las imágenes')
    for fileName in os.listdir(personPath):
        labels.append(label)
        facesData.append(cv2.imread(personPath+'/'+fileName,0))
    label = label + 1

face_recognizer = cv2.face.EigenFaceRecognizer_create()

print("Entrenando...")
face_recognizer.train(facesData, np.array(labels))
face_recognizer.write('xml/entrenamientoEigenFace.xml')
print("Modelo almacenado")

tiempoFinal=time.time()
print("Tiempo de ejecución: ", tiempoFinal-tiempoInicio, "segundos")
