# Face and Speaker Recognition on Jetson Nano
This repository documents a facial recognition system implemented on an NVIDIA Jetson Nano. To complement the recognition system, a speaker recognition algorithm is developed. The Jetson Nano is used to be able to send outputs to IoT devices when recognising a person.

## Datasets
Se han escogido dos bases de datos para hacer las pruebas de Face Detection y Face Recognition con imágenes.

### LFW
Labeled Faces in the Wild es una base de datos de fotografías de caras de celebridades diseñada para estudiar el problema del reconocimiento facial sin restricciones. El conjunto de datos contiene más de 13.000 imágenes de caras recogidas de la web y cada rostro está etiquetado con el nombre de la persona. Los rostros han sido detectados por el detector de rostros Viola-Jones.

LFW contiene cuatro conjuntos diferentes de imágenes, y se ha escogido LFW-a, conjunto en el cual se ha aplicado un método de alineación a las imágenes para facilitar el proceso de detección y reconocimiento.

### UTK Face
Este dataset es un conjunto de datos de rostros a gran escala con un amplio rango de edad (entre 0 y 116 años). El conjunto de datos consta de más de 20.000 imágenes faciales con anotaciones de edad, sexo y origen étnico en el nombre de cada fotografía. Las imágenes cubren una gran variación en cuanto a pose, expresión facial, iluminación, oclusión, resolución, etc. A diferencia de la base de datos anterior, las caras fotografiadas de UTK Face son de personas anónimas, y el objetivo es hacer pruebas con la gran variedad de expresiones, etnias y edades de las personas para comprobar hasta dónde es capaz de llegar el detector de rostros.

## first_tests.py
In this file, face detection and recognition is carried out with the OpenCV and dlib libraries. These are the first tests carried out in the work, so the oldest techniques have been used, which are Haar Cascades (Viola-Jones) and HOG for detection and EigenFaces, Fisherfaces and LBP for recognition.

### Face Detection:
- Haar Cascades (Viola-Jones): This algorithm finds the relevant features of the face from the difference between the sum of the pixels in different regions of the image, the main advantage that has been experienced with this algorithm is its extremely fast detection speed. On the other hand, it is prone to false positives, selecting faces in areas of the image where no face is located. In addition, it is generally less accurate than face detectors based on deep learning.
- HOG: A feature descriptor that counts gradient orientation occurrences in the localised part of an image. This method is more accurate than the previous one, but computationally it is significantly slower.
<p float="left">
  <img src="https://user-images.githubusercontent.com/92673739/151597832-ab7fc224-ca8b-4c1d-b80b-d1a00974cb28.jpg" width="500"/>
  <img src="https://user-images.githubusercontent.com/92673739/151597879-a586a7f5-7a40-4cd1-b926-17c80a8e5217.jpg" width="500"/>
</p>


### Reconocimiento facial:
- EigenFaces: Es el método de reconocimiento facial más antiguo y uno de los más simples. Funciona de forma bastante fiable en la mayoría de los entornos controlados. Se obtienen las siguientes conclusiones:
  - Es relativamente rápido.
  - No es lo suficientemente preciso por sí mismo y necesita métodos de impulso para su mejora.
- FisherFaces: es una técnica similar a Eigenfaces pero está orientada a mejorar la agrupación de clases. Mientras que Eigenfaces se basa en PCA, Fischer faces se basa en LDA. Se obtienen las siguientes conclusiones:
  - Produce un mejor rendimiento que EigenFaces.
  - Pierde la capacidad de reconstruir las caras.
- LBP: Es un operador de textura simple pero muy eficiente que etiqueta los píxeles de una imagen mediante el umbral de la vecindad de cada píxel y codifica el resultado como un número binario. Se obtienen las siguientes conclusiones:
  - Puede representar características locales en las imágenes.
  - Es posible obtener grandes resultados principalmente en un entorno controlado.
  - Es robusto frente a transformaciones monótonas de escala de grises.

A pesar de las ventajas de cada algoritmo de reconocimiento, ninguno de ellos aporta los resultados necesarios para el proyecto, ya que el reconocimiento suele dar erróneo cuando el programa se ejecuta con imágenes de test fotografiadas en entornos distintos que las imágenes con las que se ha entrenado el modelo.


## DetectFaces.py y DetectFacesVideo.py
Estos dos archivos contienen código casi idéntico, con la diferencia de que uno es para detectar rostros en imágenes y el otro en vídeos, streamings y cámaras externas. Para ello, se utiliza OpenCV, que tiene un módulo DNN (Deep Neural Network) que permite cargar redes neuronales pre-entrenadas; esto mejora increíblemente la velocidad, reduce la necesidad de dependencias y la mayoría de los modelos tienen un tamaño muy ligero. Se utiliza un modelo preentrenado de Face Detection que permite localizar la cara a partir de una imagen dada.

Para la ejecución del código, serán necesarios dos tipos de archivos:
- opencv_face_detector_uint8.pb: contiene el modelo.
- opencv_face_detector.pbtxt: contiene la configuración para el modelo anterior.

Para el proceso se hace uso de una BLOB (Binary Large Object), que es un elemento que se utiliza para almacenar datos de gran tamaño que cambian de forma dinámica; en este caso, la función *dnn.BlobFromImage* se encarga del preprocesamiento, que incluye la configuración de las dimensiones de la blob (se usa de entrada para la imagen) y la normalización.
Después, se pasa la blob por la red con *net.setInput(blob)* y se obtienen las detecciones y las predicciones con *net.forward()*. A partir de ahí se hace un bucle sobre las detecciones y se dibujan recuadros alrededor de las caras detectadas.

En el bucle, primeramente, se extrae la confianza (probabilidad) y se compara con el threshold de confianza, para así filtrar las detecciones débiles. Si cumple la condición anterior, se calculan las coordenadas que delimitan el rectángulo del rostro detectado. Se dibuja el rectángulo alrededor del rostro detectado y se muestra el porcentaje de confianza sobre ese rostro, lo cual facilitará mucho las pruebas de detección.

Las conclusiones obtenidas son muy buenas, ya que el programa se ejecuta con rapidez y realiza la detección facial mucho mejor que los métodos empleados hasta el momento, proporcionando mucha más fiabilidad con los falsos positivos y detectando caras orientadas en diferentes ángulos de forma correcta.

## face_recognition.py
This file contains the code from which face recognition is performed.

Face recognition in the project is done by FaceNet, a system that uses a deep convolutional network. The network is pre-trained through a triple loss function, which encourages vectors of the same person to become more similar (smaller distance) and those of different individuals to become less similar (larger distance).
The generalised operation of the system consists of transforming each face in the database into a 128-feature vector, which is called embedding. For each entry, the same transformation is applied to the detected faces and their identity with the most similar embedding in the database is predicted (as long as the difference is not greater than a verification threshold).


Los métodos utilizados anteriormente para el reconocimiento de caras implican la necesidad de grandes datos para una sola persona y un tiempo de entrenamiento para cada nueva adición al conjunto de datos. Sin embargo, la mayoría de las técnicas modernas de reconocimiento facial utilizan una alternativa, denominada One-Shot Learning. Este aprendizaje consiste en encontrar la mejor coincidencia del input con los casos de entrenamiento disponibles, en lugar de intentar clasificar la imagen de prueba con un modelo entrenado. Se trata de utilizar una imagen de entrenamiento por clase y comparar la imagen de prueba con todas las imágenes de entrenamiento. Para ello, se utiliza FaceNet: A Unified Embedding for Face Recognition and Clustering, para generar los embeddings y compararlos como sugiere Siamese Neural Networks for One-shot Image Recognition.

El código contiene las siguientes funciones:

  - **load_opencv():** La función toma el directorio del modelo .pb congelado y un archivo .pbtxt para inicializar el DNN Face Detector de OpenCV, explicado en los archivos anteriores.
  - **load_face_detection():** La función genera (a partir del modelo de Face Detection) un vector de embeddings para una imagen determinada. Estos embeddings consisten en características dentro de la imagen; los datos de la imagen se convierten en datos numéricos que pueden ser utilizados para fines de comparación. El modelo de detección de caras está en forma de gráfico Tensorflow y el archivo *_face_detection.py* contiene las funciones para cargar el modelo directamente desde el directorio. Una vez cargado el modelo, lo inicializamos con los valores por defecto.
  - **detect_faces(image):** La función toma una imagen como entrada y devuelve una lista que contiene las coordenadas de las caras dentro de la imagen. El proceso completo está explicado en el apartado anterior. En caso de que el modelo no detecte correctamente las imágenes, se puede cambiar el valor del threshold, reduciéndolo si el modelo no es capaz de detectar una cara válida o aumentándolo si detecta otros objetos como cara o detecta caras superpuestas.
  - **load_face_embeddings(image_dir):** Función que toma el directorio de las imágenes. Un bucle recorre todas las imágenes del directorio, detecta la cara en la imagen con detect_faces y guarda su embedding en el diccionario de embeddings.
  - **is_same(emb1, emb2):** Compara dos matrices y devuelve la diferencia entre ellas como un valor escalar. La función recibe dos embeddings, correspondientes a las características de dos rostros, y los compara. Retorna un valor booleano determinando si los embeddings se asemejan lo suficiente como para ser la misma persona, utilizando la variable *verification_threshold*. Esta variable se puede modificar según las necesidades; si se encuentra un rostro pero no se reconoce como se esperaba, se aumenta el threshold, y si se reconoce un rostro como persona equivocada, se disminuye.
  - **fetch_detections(image, embeddings):** Después de tener todos los modelos cargados y los embeddings para las imágenes de entrenamiento calculadas, se puede ejecutar el reconocimiento facial para la imagen entrada. En primer lugar, se detectan las caras dentro de la imagen utilizando detect_faces() y se encuentra su embedding. A continuación, se compara el embedding de la imagen de prueba con cada embedding de la imagen de entrenamiento. Si hay varias detecciones, se ordenan según las diferencias y se asigna la imagen con la menor diferencia con la imagen detectada. Como puede haber más de una cara en la imagen de entrada, el parámetro de detecciones es una matriz.
  - **face_recognition(image_or_video_path, Bool : display_image, String : face_dir):** Ahora que están definidas todas las funciones, se escribe esta función para envolver todo el proceso. Esta función se encarga de los parámetros, carga los modelos, embeddings, maneja el cambio de imagen, vídeo y webcam y ejecuta la detección basada en la entrada. Se puede llamar a esta función usando un archivo *__main__* que toma los argumentos de la consola y los envía a la función.

El código de face_recognition.py y *_face_detection.py* ha sido extraído del repositorio https://github.com/deepme987/face-recognition-python

