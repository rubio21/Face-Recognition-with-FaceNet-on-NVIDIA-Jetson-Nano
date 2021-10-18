# TFG_FaceRecognition
Este repositorio recoge las pruebas realizadas durante la duración de mi Trabajo de Fin de Grado, el cual está basado en el reconocimiento facial. Se profundizará con diferentes técnicas de detección, procesamiento y reconocimiento del rostro a lo largo del proyecto. El repositario se actualizará iterativamente con nuevos ficheros y se explicarán las conclusiones obtenidas en el correspondiente README.

## FirstTests.py
En este fichero se realiza la detección y el reconocimiento facial con las librerías OpenCV y dlib. Han sido las primeras pruebas realizadas en el trabajo, por lo que se han utilizado las técnicas más antiguas, que son Haar Cascades (Viola-Jones) y HOG para la detección y EigenFaces, Fisherfaces y LBP para el reconocimiento.
- Detección facial:
  - Haar Cascades (Viola-Jones): La principal ventaja que se ha experimentado con este algoritmo es su extremada rapidez para realizar la detección. Por contra, es propenso a los falsos positivos, seleccionando rostros en zonas de la imagen donde no se ubica ninguna cara. Además, en general, es menos preciso que los detectores de rostros basados en deep learning.
  - HOG: Este método es más preciso que el anterior, pero computacionalmente es significativamente más lento.

- Reconocimiento facial:
  - 
