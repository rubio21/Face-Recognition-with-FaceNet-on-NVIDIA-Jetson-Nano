# TFG_FaceRecognition
Este repositorio recoge las pruebas realizadas durante la duración de mi Trabajo de Fin de Grado, el cual está basado en el reconocimiento facial. Se profundizará con diferentes técnicas de detección, procesamiento y reconocimiento del rostro a lo largo del proyecto. El repositorio se actualizará iterativamente con nuevos ficheros y se explicarán las conclusiones obtenidas en el correspondiente README.

## Datasets
Se han escogido dos bases de datos para hacer las pruebas de Face Detection y Face Recognition con imágenes.

### LFW
Labeled Faces in the Wild es una base de datos de fotografías de caras de celebridades diseñada para estudiar el problema del reconocimiento facial sin restricciones. El conjunto de datos contiene más de 13.000 imágenes de caras recogidas de la web y cada rostro está etiquetado con el nombre de la persona. Los rostros han sido detectados por el detector de rostros Viola-Jones.

LFW contiene cuatro conjuntos diferentes de imágenes, y se ha escogido LFW-a, conjunto en el cual se ha aplicado un método de alineación a las imágenes para facilitar el proceso de detección y reconocimiento.

### UTK Face
Este dataset es un conjunto de datos de rostros a gran escala con un amplio rango de edad (entre 0 y 116 años). El conjunto de datos consta de más de 20.000 imágenes faciales con anotaciones de edad, sexo y origen étnico en el nombre de cada fotografía. Las imágenes cubren una gran variación en cuanto a pose, expresión facial, iluminación, oclusión, resolución, etc. A diferencia de la base de datos anterior, las caras fotografiadas de UTK Face son de personas anónimas, y el objetivo es hacer pruebas con la gran variedad de expresiones, etnias y edades de las personas para comprobar hasta dónde es capaz de llegar el detector de rostros.

## FirstTests.py
En este fichero se realiza la detección y el reconocimiento facial con las librerías OpenCV y dlib. Han sido las primeras pruebas realizadas en el trabajo, por lo que se han utilizado las técnicas más antiguas, que son Haar Cascades (Viola-Jones) y HOG para la detección y EigenFaces, Fisherfaces y LBP para el reconocimiento.
### Detección facial:
- Haar Cascades (Viola-Jones): Este algoritmo encuentra las características relevantes del rostro a partir de la diferencia entre la suma de los píxeles de diferentes regiones de la imagen.La principal ventaja que se ha experimentado con este algoritmo es su extremada rapidez para realizar la detección. Por contra, es propenso a los falsos positivos, seleccionando rostros en zonas de la imagen donde no se ubica ninguna cara. Además, en general, es menos preciso que los detectores de rostros basados en deep learning.
- HOG: Es un descriptor de características que cuenta las incidencias de orientación de gradiente en la parte localizada de una imagen. Este método es más preciso que el anterior, pero computacionalmente es significativamente más lento.
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

## FaceRecognition.py
Este fichero contiene el código a partir del cual se realiza el reconocimiento facial, con todas las funcionalidades necesarias para reconocer a personas a partir de imágenes o vídeos.

Los métodos utilizados anteriormente para el reconocimiento de caras implican la necesidad de grandes datos para una sola persona y un tiempo de entrenamiento para cada nueva adición al conjunto de datos. Sin embargo, la mayoría de las técnicas modernas de reconocimiento facial utilizan una alternativa, denominada One-Shot Learning. Este aprendizaje consiste en encontrar la mejor coincidencia del input con los casos de entrenamiento disponibles, en lugar de intentar clasificar la imagen de prueba con un modelo entrenado. Se trata de utilizar una imagen de entrenamiento por clase y comparar la imagen de prueba con todas las imágenes de entrenamiento. Para ello, se utiliza FaceNet: A Unified Embedding for Face Recognition and Clustering, para generar los embeddings y compararlos como sugiere Siamese Neural Networks for One-shot Image Recognition.

El código contiene las siguientes funciones:

  - **load_opencv():** La función toma el directorio del modelo .pb congelado y un archivo .pbtxt para inicializar el DNN Face Detector de OpenCV, explicado en los archivos anteriores.
  - **load_face_detection():** La función genera (a partir del modelo de Face Detection) un vector de embeddings para una imagen determinada. Estos embeddings consisten en características dentro de la imagen; los datos de la imagen se convierten en datos numéricos que pueden ser utilizados para fines de comparación. El modelo de detección de caras está en forma de gráfico Tensorflow y el archivo *_face_detection.py* contiene las funciones para cargar el modelo directamente desde el directorio. Una vez cargado el modelo, lo inicializamos con los valores por defecto.
  - **detect_faces(image):** La función toma una imagen como entrada y devuelve una lista que contiene las coordenadas de las caras dentro de la imagen. El proceso completo está explicado en el apartado anterior. En caso de que el modelo no detecte correctamente las imágenes, se puede cambiar el valor del threshold, reduciéndolo si el modelo no es capaz de detectar una cara válida o aumentándolo si detecta otros objetos como cara o detecta caras superpuestas.
  - **load_face_embeddings(image_dir):** Función que toma el directorio de las imágenes. Un bucle recorre todas las imágenes del directorio, detecta la cara en la imagen con detect_faces y guarda su embedding en el diccionario de embeddings.
  - **is_same(emb1, emb2):** Compara dos matrices y devuelve la diferencia entre ellas como un valor escalar. La función recibe dos embeddings, correspondientes a las características de dos rostros, y los compara. Retorna un valor booleano determinando si los embeddings se asemejan lo suficiente como para ser la misma persona, utilizando la variable *verification_threshold*. Esta variable se puede modificar según las necesidades; si se encuentra un rostro pero no se reconoce como se esperaba, se aumenta el threshold, y si se reconoce un rostro como persona equivocada, se disminuye.
  - **fetch_detections(image, embeddings):** Después de tener todos los modelos cargados y los embeddings para las imágenes de entrenamiento calculadas, se puede ejecutar el reconocimiento facial para la imagen entrada. En primer lugar, se detectan las caras dentro de la imagen utilizando detect_faces() y se encuentra su embedding. A continuación, se compara el embedding de la imagen de prueba con cada embedding de la imagen de entrenamiento. Si hay varias detecciones, se ordenan según las diferencias y se asigna la imagen con la menor diferencia con la imagen detectada. Como puede haber más de una cara en la imagen de entrada, el parámetro de detecciones es una matriz.
  - **face_recognition(image_or_video_path, Bool : display_image, String : face_dir):** Ahora que están definidas todas las funciones, se escribe esta función para envolver todo el proceso. Esta función se encarga de los parámetros, carga los modelos, embeddings, maneja el cambio de imagen, vídeo y webcam y ejecuta la detección basada en la entrada. Se puede llamar a esta función usando un archivo *__main__* que toma los argumentos de la consola y los envía a la función.
