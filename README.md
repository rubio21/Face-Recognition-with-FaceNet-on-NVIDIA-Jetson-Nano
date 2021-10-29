# TFG_FaceRecognition
Este repositorio recoge las pruebas realizadas durante la duración de mi Trabajo de Fin de Grado, el cual está basado en el reconocimiento facial. Se profundizará con diferentes técnicas de detección, procesamiento y reconocimiento del rostro a lo largo del proyecto. El repositario se actualizará iterativamente con nuevos ficheros y se explicarán las conclusiones obtenidas en el correspondiente README.

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
