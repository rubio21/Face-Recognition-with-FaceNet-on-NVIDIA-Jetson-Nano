# Face Recognition with FaceNet on Jetson Nano
This repository documents a facial recognition system implemented on an NVIDIA Jetson Nano. To complement the recognition system, a speaker recognition algorithm is developed. The Jetson Nano is used to be able to send outputs to IoT devices when recognising a person.

## first_tests.py
In this file, face detection and recognition is carried out with the OpenCV and dlib libraries. These are the first tests carried out in the work, so the oldest techniques have been used, which are Haar Cascades (Viola-Jones) and HOG for detection and EigenFaces, Fisherfaces and LBP for recognition.

### Face Detection:
- **Haar Cascades (Viola-Jones)**: This algorithm finds the relevant features of the face from the difference between the sum of the pixels in different regions of the image, the main advantage that has been experienced with this algorithm is its extremely fast detection speed. On the other hand, it is prone to false positives, selecting faces in areas of the image where no face is located. In addition, it is generally less accurate than face detectors based on deep learning.
- **HOG**: A feature descriptor that counts gradient orientation occurrences in the localised part of an image. This method is more accurate than the previous one, but computationally it is significantly slower.
<p float="left">
  <img src="https://user-images.githubusercontent.com/92673739/151597832-ab7fc224-ca8b-4c1d-b80b-d1a00974cb28.jpg" width="500"/>
  <img src="https://user-images.githubusercontent.com/92673739/151597879-a586a7f5-7a40-4cd1-b926-17c80a8e5217.jpg" width="500"/>
</p>

As can be seen from the results, both algorithms have correctly detected 4 out of 7 possible faces. Viola-Jones provided 5 false positives and HOG only 1, but Viola-Jones ran significantly faster. Both methods have avoided the same faces, two of which are wearing sunglasses and one of which is facing away from the camera.

### Face recognition:
- **EigenFaces**
- **FisherFaces**
- **LBPH**

The OpenCV library has been used in the experiments carried out. A model has been created for each of the three face recognition procedures described above, based on previously detected faces. For the detected face, a prediction of the most similar being from the training set is generated, associated with a confidence factor. This factor is different for each procedure, so a threshold (thres-hold) has been used to classify a prediction as "unknown" in case it is exceeded, being 4500 in Eigenfaces, 450 in Fisherfaces and 70 in LBP.
The three options coincide in frequent recognition errors in images with an environment and illumination unequal to those of the training set. In addition, numerous samples of each person are needed to train the patterns, so it takes too much memory and the model takes too long to create. Therefore, these techniques do not provide the necessary reliability and speed.



## detect_faces.py
This file detects faces in an image or video from a Deep Neural Network. The network is used through OpenCV, which includes a DNN module that allows pre-trained neural networks to be loaded; this greatly improves speed, reduces the need for dependencies and most models are very light in size. The neural network used is based on **Single-Shot Detector (SSD)** with a ResNet base network.
Single-Shot Detector (SSD) is an algorithm that detects the object (face) in a single pass over the input image, unlike other models that traverse it more than once. SSD is based on the use of convolutional networks that produce multiple bounding boxes of various fixed sizes and score the presence of the object in those boxes, followed by a suppression step to produce the final detections.

In the code implementation, the pre-trained model is read through two files and the test images are fed into the network, which returns the detections found. The files are:
- **opencv_face_detector_uint8.pb**: contains the model.
- **opencv_face_detector.pbtxt**: contains the configuration for the model.

The process makes use of a BLOB (Binary Large Object), which is an element used to store dynamically changing large data; in this case, the *dnn.BlobFromImage* function handles the preprocessing, which includes setting the dimensions of the blob (used as input for the image) and normalisation.
Then, the blob is passed through the network with *net.setInput(blob)* and detections and predictions are obtained with *net.forward()*. For greater precision, the detections obtained are run through and their associated confidence value is compared with a threshold, in order to eliminate the least reliable ones.

The results obtained have been very good, since an algorithm that greatly improves Viola Jones and HOG is achieved, showing itself to be accurate and robust in the detection of faces with different viewing angles and lighting and occlusion conditions.

<img src="https://user-images.githubusercontent.com/92673739/151672189-fd33fb34-d2ff-4eb2-a6e8-aa611f2a95ef.jpg" width="500"/>

In this example, an accuracy of 100% is obtained. Unlike the previous methods, this algorithm has been able to detect both faces with glasses and the face from the side of the camera.

This file is not executed in the main program, which is *face_recognition.py*. It is convenient to use it when you want to test only the detector, or when you want to save a new person in the database. The program captures the best confidence image provided by SSD from a video or stream and stores it in the faces folder.
## face_recognition.py
This file contains the code from which face detection and face recognition in real time are performed.

Previously used methods for face recognition involve the need for large data for a single person and a training time for each new addition to the dataset. To avoid this drawback we use One-Shot Learning, a computer vision technique that allows the learning of information about faces from a single image, and without the need to re-train the model.
Face recognition in the project is done by FaceNet, a system that uses a deep convolutional network. The network is pre-trained through a triple loss function, which encourages vectors of the same person to become more similar (smaller distance) and those of different individuals to become less similar (larger distance).
The generalised operation of the system consists of transforming each face in the database into a 128-feature vector, which is called embedding. For each entry, the same transformation is applied to the detected faces and their identity with the most similar embedding in the database is predicted (as long as the difference is not greater than a verification threshold).

The files containing the pre-trained neural networks are located in the 'FaceDetection/' and 'FaceRecognition/' folders of 'Models/'.


El código contiene las siguientes funciones:

  - **load_opencv():** La función toma el directorio del modelo .pb congelado y un archivo .pbtxt para inicializar el DNN Face Detector de OpenCV, explicado en los archivos anteriores.
  - **load_face_detection():** La función genera (a partir del modelo de Face Detection) un vector de embeddings para una imagen determinada. Estos embeddings consisten en características dentro de la imagen; los datos de la imagen se convierten en datos numéricos que pueden ser utilizados para fines de comparación. El modelo de detección de caras está en forma de gráfico Tensorflow y el archivo *_face_detection.py* contiene las funciones para cargar el modelo directamente desde el directorio. Una vez cargado el modelo, lo inicializamos con los valores por defecto.
  - **detect_faces(image):** La función toma una imagen como entrada y devuelve una lista que contiene las coordenadas de las caras dentro de la imagen. El proceso completo está explicado en el apartado anterior. En caso de que el modelo no detecte correctamente las imágenes, se puede cambiar el valor del threshold, reduciéndolo si el modelo no es capaz de detectar una cara válida o aumentándolo si detecta otros objetos como cara o detecta caras superpuestas.
  - **load_face_embeddings(image_dir):** Función que toma el directorio de las imágenes. Un bucle recorre todas las imágenes del directorio, detecta la cara en la imagen con detect_faces y guarda su embedding en el diccionario de embeddings.
  - **is_same(emb1, emb2):** Compara dos matrices y devuelve la diferencia entre ellas como un valor escalar. La función recibe dos embeddings, correspondientes a las características de dos rostros, y los compara. Retorna un valor booleano determinando si los embeddings se asemejan lo suficiente como para ser la misma persona, utilizando la variable *verification_threshold*. Esta variable se puede modificar según las necesidades; si se encuentra un rostro pero no se reconoce como se esperaba, se aumenta el threshold, y si se reconoce un rostro como persona equivocada, se disminuye.
  - **fetch_detections(image, embeddings):** Después de tener todos los modelos cargados y los embeddings para las imágenes de entrenamiento calculadas, se puede ejecutar el reconocimiento facial para la imagen entrada. En primer lugar, se detectan las caras dentro de la imagen utilizando detect_faces() y se encuentra su embedding. A continuación, se compara el embedding de la imagen de prueba con cada embedding de la imagen de entrenamiento. Si hay varias detecciones, se ordenan según las diferencias y se asigna la imagen con la menor diferencia con la imagen detectada. Como puede haber más de una cara en la imagen de entrada, el parámetro de detecciones es una matriz.
  - **face_recognition(image_or_video_path, Bool : display_image, String : face_dir):** Ahora que están definidas todas las funciones, se escribe esta función para envolver todo el proceso. Esta función se encarga de los parámetros, carga los modelos, embeddings, maneja el cambio de imagen, vídeo y webcam y ejecuta la detección basada en la entrada. Se puede llamar a esta función usando un archivo *__main__* que toma los argumentos de la consola y los envía a la función.

El código de face_recognition.py y *_face_detection.py* ha sido extraído del repositorio https://github.com/deepme987/face-recognition-python

