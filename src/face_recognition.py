
import os
import cv2
import argparse
import numpy as np

import _face_detection as ftk


class FaceDetection:
    verification_threshold = 0.8
    v, net = None, None
    image_size = 160

    def __init__(self):
        FaceDetection.load_models()

    @staticmethod
    # Carga de los modelos
    def load_models():
        if not FaceDetection.net:
            FaceDetection.net = FaceDetection.load_opencv()

        if not FaceDetection.v:
            FaceDetection.v = FaceDetection.load_face_detection()
        
    @staticmethod
    # OpenCV DNN Face Detector: la función toma el directorio del modelo .pb congelado y un archivo .pbtxt.
    def load_opencv():
        model_path = "../Models/OpenCV/opencv_face_detector_uint8.pb"
        model_pbtxt = "../Models/OpenCV/opencv_face_detector.pbtxt"

        # Se carga nuestro modelo serializado desde el disco
        net = cv2.dnn.readNetFromTensorflow(model_path, model_pbtxt)
        return net

    @staticmethod
    # Generación (a partir del modelo de Face Detection) un vector de embeddings para una imagen determinada.
    def load_face_detection():
        v = ftk.Verification()
        v.load_model("../Models/FaceDetection/")
        # Inicialización con los valores por defecto.
        v.initial_input_output_tensors()
        return v

    @staticmethod
    # Compara dos matrices y devuelve la diferencia entre ellas como un valor escalar.
    def is_same(emb1, emb2): # Input: dos embeddings correspondientes a las características de dos rostros.
        diff = np.subtract(emb1, emb2)
        diff = np.sum(np.square(diff))
        return diff < FaceDetection.verification_threshold, diff

    @staticmethod
    # Detección de caras en imagen o vídeo
    def detect_faces(image, display_images=False): # Poner display_image = True para depurar manualmente si te encuentras con errores
        height, width, channels = image.shape
        # Se carga la imagen de entrada y se construye un blob de entrada para la imagen,
        # cambiando el tamaño a un valor fijo de 300x300 píxeles y normalizándolo
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
        # Se pasa la blob por la red y se obtienen las detecciones y predicciones
        FaceDetection.net.setInput(blob)
        detections = FaceDetection.net.forward()

        faces = []

        # Bucle sobre las detecciones
        for i in range(detections.shape[2]):
            # Se extrae la confianza (probabilidad) asociada a la predicción
            confidence = detections[0, 0, i, 2]
            # Se filtran las detecciones débiles, comprobando que la confianza es mayor que el threshold
            if confidence > 0.5:
                # Se calculan las coordenadas que delimitan el rectángulo del rostro
                x1 = int(detections[0, 0, i, 3] * width)
                y1 = int(detections[0, 0, i, 4] * height)
                x2 = int(detections[0, 0, i, 5] * width)
                y2 = int(detections[0, 0, i, 6] * height)
                faces.append([x1, y1, x2 - x1, y2 - y1])

                if display_images:
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
    
        if display_images:
            print("Face co-ordinates: ", faces)
            cv2.imshow("Training Face", cv2.resize(image, (300, 300)))
            cv2.waitKey(0)
        return faces

    @staticmethod
    # Almacenamiento de embeddings en un diccionario. Input: directorio de las imágenes
    def load_face_embeddings(image_dir):

        embeddings = {}
        # Bucle para recorrer todas las imágenes del directorio
        for file in os.listdir(image_dir):
            img_path = image_dir + file
            try:
                image = cv2.imread(img_path)
                # Detección de caras
                faces = FaceDetection.detect_faces(image)
                if len(faces) == 1:
                    x, y, w, h = faces[0]
                    image = image[y:y + h, x:x + w]
                    # Guardar embedding en el diccionario de embeddings
                    embeddings[file.split(".")[0]] = FaceDetection.v.img_to_encoding(cv2.resize(image, (160, 160)), FaceDetection.image_size)
                else:
                    print(f"Found more than 1 face in \"{file}\", skipping embeddings for the image.")
            except Exception:
                print(f"Unable to read file: {file}")
        return embeddings

    @staticmethod
    # Reconocimiento facial
    def fetch_detections(image, embeddings, display_image_with_detections=False):
        # Detección de caras en la imagen
        faces = FaceDetection.detect_faces(image)
        
        detections = []
        for face in faces:
            x, y, w, h = face
            im_face = image[y:y + h, x:x + w]
            img = cv2.resize(im_face, (200, 200))
            # Obtención de embeddings
            user_embed = FaceDetection.v.img_to_encoding(cv2.resize(img, (160, 160)), FaceDetection.image_size)
            
            detected = {}
            for _user in embeddings:
                # Comparación de las características de la imagen con todas las almacenadas previamente
                flag, thresh = FaceDetection.is_same(embeddings[_user], user_embed)
                # Se guardan los rostros reconocidos junto a la diferencia respecto a la imagen más cercana
                if flag:
                    detected[_user] = thresh
            # Ordenación a partir de la diferencia
            detected = {k: v for k, v in sorted(detected.items(), key=lambda item: item[1])}
            detected = list(detected.keys())
            if len(detected) > 0:
                detections.append(detected[0])
                # Se dibuja un rectángulo alrededor del rostro junto al nombre de la persona reconocida
                if display_image_with_detections:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(image, detected[0], (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        # Se muestra si se ha especificado por parámetro
        if display_image_with_detections:
            cv2.imshow("Detected", cv2.resize(image, (300, 300)))

        return detections

# Se envuelve todo en una función
def face_recognition(image_or_video_path=None, display_image=False, face_dir="../faces/"):
    # Se cargan los modelos
    FaceDetection.load_models()
    # Se almacenan los embeddings de las imágenes del dataset
    embeddings = FaceDetection.load_face_embeddings(face_dir)
    waitkey_variable = 1
    image_flip = False
    # Obtención por imagen o vídeo entrados
    if image_or_video_path:
        print("Using path: ", image_or_video_path)
        cap = cv2.VideoCapture(image_or_video_path)
        if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 1:
            waitkey_variable = 0
    # Obtención por streaming
    else:
        print("Capturing from webcam")
        image_flip = True
        cap = cv2.VideoCapture(0)
    
    while 1:
        ret, image = cap.read()
        if image_flip:
            image = cv2.flip(image, 1)

        if not ret:
            print("Finished detection")
            return
        # Reconocimiento facial
        print(FaceDetection.fetch_detections(image, embeddings, display_image))

        key = cv2.waitKey(waitkey_variable)
        if key & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None, help='Path to input file')
    parser.add_argument("--display-image", action="store_true", help="Display Detected Image")
    parser.add_argument('--faces-dir', type=str, default="../faces/", help='Path to faces dir')
    args = parser.parse_args()

    face_recognition(args.input, args.display_image, args.faces_dir)
