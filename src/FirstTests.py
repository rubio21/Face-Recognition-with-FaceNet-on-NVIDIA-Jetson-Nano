import cv2
import os
import numpy as np
import dlib


peopleList = os.listdir('lfw2/lfw2/')[:8]
print('Lista de personas: ', peopleList)
faceClassif = cv2.CascadeClassifier("xml/haarcascade_frontalface_default.xml")
labels = []
facesData = []
label = 0

for nameDir in peopleList:
    if not os.path.exists('Data/' + nameDir):
        print('Carpeta creada: ', nameDir)
        os.makedirs('Data/' + nameDir)
    count=0
    personPath = 'lfw2/lfw2/' + nameDir
    print('Leyendo las imágenes de ' + nameDir)
    for fileName in os.listdir(personPath):
        image = cv2.imread(personPath+'/'+fileName,0)
        auxImage = image.copy()

        # Viola-Jones
        faces = faceClassif.detectMultiScale(image, 1.3, 5)

        # HOG
        # hogFaceDetector = dlib.get_frontal_face_detector()
        # faces = hogFaceDetector(image, 1)

        # Viola-Jones
        for (x, y, w, h) in faces:

        # HOG
        # for (i, rect) in enumerate(faces):
        #     x = rect.left()
        #     y = rect.top()
        #     w = rect.right() - x
        #     h = rect.bottom() - y


            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            rostro = auxImage[y:y + h, x:x + w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite('Data/' + nameDir + '/rostro_{}.jpg'.format(count), rostro)
            count = count + 1
            labels.append(label)
            facesData.append(rostro)
        cv2.imshow('frame', image)
        k = cv2.waitKey(1)
    label=label+1



### TRAIN
# Eigenfaces
# face_recognizer = cv2.face.EigenFaceRecognizer_create()

# FisherFaces
# face_recognizer = cv2.face.FisherFaceRecognizer_create()

# LBPH
face_recognizer = cv2.face.LBPHFaceRecognizer_create()



print("Entrenando...")
face_recognizer.train(facesData, np.array(labels))
face_recognizer.write('xml/entrenamiento.xml')
print("Modelo almacenado")


### TEST
imagePaths = os.listdir('Data/')
print('imagePaths=', imagePaths)
nombreTest='Aaron_Guiel'
imagenTest=cv2.imread('Test/' + nombreTest + '.jpg')
gray = cv2.cvtColor(imagenTest, cv2.COLOR_BGR2GRAY)
auxFrame = gray.copy()
faces = faceClassif.detectMultiScale(gray, 1.3, 5)
for (x, y, w, h) in faces:
    rostro = auxFrame[y:y + h, x:x + w]
    rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
    result = face_recognizer.predict(rostro)
    cv2.putText(imagenTest, '{}'.format(result), (x, y - 5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

    cv2.putText(imagenTest, '{}'.format(imagePaths[result[0]]), (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.rectangle(imagenTest, (x, y), (x + w, y + h), (0, 255, 0), 2)




cv2.imshow(nombreTest, imagenTest)
k = cv2.waitKey(0)
if k==27:
    cv2.destroyAllWindows()

