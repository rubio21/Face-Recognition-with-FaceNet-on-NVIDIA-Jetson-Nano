# TFG_FaceRecognition
Este repositorio recoge las pruebas realizadas durante la duración de mi Trabajo de Fin de Grado, el cual está basado en el reconocimiento facial. Se profundizará con diferentes técnicas de detección, procesamiento y reconocimiento del rostro a lo largo del proyecto. El repositario se actualizará iterativamente con nuevos ficheros y se explicarán las conclusiones obtenidas en el correspondiente README.

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


## DetectFaces.py y DetectFaces_Video.py
Estos dos archivos contienen código casi idéntico, con la diferencia de que uno es para detectar rostros en imágenes y el otro en vídeos, streamings y cámaras externas. Para ello, se utiliza Deep Learning con OpenCV. En este caso, se utiliza un modelo entrenado con Caffe, un marco de aprendizaje profundo creado pensando en la expresión, la velocidad y la modularidad. El código está basado en la explicación de Adrian Rosebrock, en su artículo "Face Detection with OpenCV and deep learning". El detector facial de deep learning de OpenCV se basa en el marco del Single-Shot Detector (SSD) con una red base ResNet.

Para la ejecución del código, serán necesarios dos tipos de archivos, que deberán especificarse en los parámetros del mismo:
- .protxt: define la arquitectura del modelo (es decir, las propias capas que forman la Deep Neural Network). Para este proyecto se ha usado el documento deploy.prototxt.txt, extraído del repositorio https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector.
- .caffemodel: contiene los pesos de las capas de la red neuronal. Para este documento se ha usado el documento res10_300x300_ssd_iter_140000.caffemodel, extraído del repositorio https://github.com/gopinath-balu/computer_vision/blob/master/CAFFE_DNN.

Los resultados del programa proporcionan un enmarque de las caras detectadas en el input, ya sea una imágen o un vídeo, con su respectivo porcentaje de confianza, que indica la seguridad del programa sobre si el rostro detectado es realmente una cara.

Las conclusiones obtenidas son muy buenas, ya que el programa se ejecuta con rapidez y realiza la detección facial mucho mejor que los métodos empleados hasta el momento, proporcionando mucha más fiabilidad con los falsos positivos y detectando caras orientadas en diferentes ángulos de forma correcta.
