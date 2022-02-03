# Face Recognition with FaceNet on Jetson Nano
This repository documents a facial recognition system implemented on an NVIDIA Jetson Nano. To complement the recognition system, a speaker recognition algorithm is developed. The Jetson Nano is used to be able to send outputs to IoT devices when recognising a person. In my case, I use the Logitech HD Pro C920 camera to obtain the images and sounds and two LEDs to simulate an output when a recognition is obtained.
The programme can also be used on your personal computer, obtaining the images via the integrated webcam and without the need to illuminate LEDs.

<p align="center">
  <img src="https://user-images.githubusercontent.com/92673739/152345376-8a5bf484-c6e1-4eec-87f3-931892c8a8c9.jpg" width="300"/>
</p>

## Face Recognition
The experiments are divided into face detection and face recognition algorithms. Both are started with older techniques (Viola-Jones and HOG in face detection and Eigenfaces, Fisherfaces and LBPH in face recognition). After obtaining and comparing the results, Single-Shot Detector and FaceNet are used for the final programme.

### first_tests.py
In this file, face detection and recognition is carried out with the OpenCV and dlib libraries. These are the first tests carried out in the work, so the oldest techniques have been used, which are Haar Cascades (Viola-Jones) and HOG for detection and EigenFaces, Fisherfaces and LBP for recognition.

#### Face Detection:
- **Haar Cascades (Viola-Jones)**: This algorithm finds the relevant features of the face from the difference between the sum of the pixels in different regions of the image, the main advantage that has been experienced with this algorithm is its extremely fast detection speed. On the other hand, it is prone to false positives, selecting faces in areas of the image where no face is located. In addition, it is generally less accurate than face detectors based on deep learning.
- **HOG**: A feature descriptor that counts gradient orientation occurrences in the localised part of an image. This method is more accurate than the previous one, but computationally it is significantly slower.
<p align="center">
  <img src="https://user-images.githubusercontent.com/92673739/151597832-ab7fc224-ca8b-4c1d-b80b-d1a00974cb28.jpg" width="500"/>
  <img src="https://user-images.githubusercontent.com/92673739/151597879-a586a7f5-7a40-4cd1-b926-17c80a8e5217.jpg" width="500"/>
</p>

As can be seen from the results, both algorithms have correctly detected 4 out of 7 possible faces. Viola-Jones provided 5 false positives and HOG only 1, but Viola-Jones ran significantly faster. Both methods have avoided the same faces, two of which are wearing sunglasses and one of which is facing away from the camera.

#### Face recognition:
- **EigenFaces**
- **FisherFaces**
- **LBPH**

The OpenCV library has been used in the experiments carried out. A model has been created for each of the three face recognition procedures described above, based on previously detected faces. For the detected face, a prediction of the most similar being from the training set is generated, associated with a confidence factor. This factor is different for each procedure, so a threshold (thres-hold) has been used to classify a prediction as "unknown" in case it is exceeded, being 4500 in Eigenfaces, 450 in Fisherfaces and 70 in LBP.
The three options coincide in frequent recognition errors in images with an environment and illumination unequal to those of the training set. In addition, numerous samples of each person are needed to train the patterns, so it takes too much memory and the model takes too long to create. Therefore, these techniques do not provide the necessary reliability and speed.



### detect_faces.py
This file detects faces in an image or video from a Deep Neural Network. The network is used through OpenCV, which includes a DNN module that allows pre-trained neural networks to be loaded; this greatly improves speed, reduces the need for dependencies and most models are very light in size. The neural network used is based on **Single-Shot Detector (SSD)** with a ResNet base network.
Single-Shot Detector (SSD) is an algorithm that detects the object (face) in a single pass over the input image, unlike other models that traverse it more than once. SSD is based on the use of convolutional networks that produce multiple bounding boxes of various fixed sizes and score the presence of the object in those boxes, followed by a suppression step to produce the final detections.

In the code implementation, the pre-trained model is read through two files and the test images are fed into the network, which returns the detections found. The files are:
- **opencv_face_detector_uint8.pb**: contains the model.
- **opencv_face_detector.pbtxt**: contains the configuration for the model.

The process makes use of a BLOB (Binary Large Object), which is an element used to store dynamically changing large data; in this case, the *dnn.BlobFromImage* function handles the preprocessing, which includes setting the dimensions of the blob (used as input for the image) and normalisation.
Then, the blob is passed through the network with *net.setInput(blob)* and detections and predictions are obtained with *net.forward()*. For greater precision, the detections obtained are run through and their associated confidence value is compared with a threshold, in order to eliminate the least reliable ones.

The results obtained have been very good, since an algorithm that greatly improves Viola Jones and HOG is achieved, showing itself to be accurate and robust in the detection of faces with different viewing angles and lighting and occlusion conditions.

<p align="center"> <img src="https://user-images.githubusercontent.com/92673739/151672189-fd33fb34-d2ff-4eb2-a6e8-aa611f2a95ef.jpg" width="500"/> </p>

In this example, an accuracy of 100% is obtained. Unlike the previous methods, this algorithm has been able to detect both faces with glasses and the face from the side of the camera.

This file is not executed in the main program, which is *face_recognition.py*. It is convenient to use it when you want to test only the detector, or when you want to save a new person in the dataset. The program captures the best confidence image provided by SSD from a video or stream and stores it in the faces folder.
### face_recognition.py
This file contains the code from which face detection and face recognition in real time are performed.

Previously used methods for face recognition involve the need for large data for a single person and a training time for each new addition to the dataset. To avoid this drawback we use **One-Shot Learning**, a computer vision technique that allows the learning of information about faces from a single image, and without the need to re-train the model.
Face recognition in the project is done by **FaceNet**, a system that uses a deep convolutional network. The network is pre-trained through a triple loss function, which encourages vectors of the same person to become more similar (smaller distance) and those of different individuals to become less similar (larger distance).
The generalised operation of the system consists of transforming each face in the dataset into a 128-feature vector, which is called embedding. For each entry, the same transformation is applied to the detected faces and their identity with the most similar embedding in the dataset is predicted (as long as the difference is not greater than a verification threshold).

Results of running face_recognition.py with *Labeled Faces in the Wild* images. The left image of each pair is the one located in the dataset and the one on the right is the test image:
<p align="center"> <img src="https://user-images.githubusercontent.com/92673739/152239732-aec9413d-c308-4de9-8f94-b73ffade57c4.png" width="500"/> </p>


The images of the persons should be included in the folder 'Dataset/' (it is important that no more than one person appears in the same photo). The files containing the pre-trained neural networks are located in the 'FaceDetection/' and 'FaceRecognition/' folders of 'Models/'. If you are not interested in the illumination of the LEDs, you have to comment the *initialise_led()* and *change_led()* functions so that there are no errors.

To run the system showing the recognitions:
<!--sec data-collapse=true ces-->

    $ python face_recognition.py --view
    
<!--endsec-->

The dataset and streaming capture paths are set by default.

If you want to change the path of the dataset or insert an input image or video:

<!--sec data-collapse=true ces-->

    $ python face_recognition.py --input 'your_file_path' --dataset 'your_dataset_path'
    
<!--endsec-->

#### Explanation of the code:

Two classes: FaceDetection and FaceRecognition.

- **FaceDetection:** Detects and calculates the location coordinates of faces located in an image with **Single-Shot Detector (SSD)**.
  - *load_face_detection()*: Load the face detection model.
  - *detect_faces()*: Detect faces in an image.
- **FaceRecognition:** It processes the faces detected by *FaceDetector* to extract their features and convert them into a 128-feature vector with **FaceNet**.
  - *load_face_recognition()*: Load the face recognition model.
  - *img_to_embedding()*: Transforms the image entered by parameter into embedding.
  - *load_face_embeddings()*: Traverses the entire dataset to transform the images into vectors and store them in a dictionary.
  - *is_same()*: Compare the difference between two vectors. The distance determines a recognition.
- *face_recognition()*: Use the two classes above to find the faces in an image and recognise them, comparing their embedding with all those in the dictionary.
- *main_program()*: Use *face_recognition()* to get recognitions throughout the video (or image).
- *initialise_led(), change_led()*: Switching the LEDs on and off.

## Considerations:

In view of the current COVID-19 situation, it is essential to wear a mask. In the case of wanting to implement the system, it must be considered that it is not able to recognise a person wearing a mask if their image in the dataset is with their face uncovered.

In order to be able to use it even with masks, the image must be added to the dataset. An experiment has been carried out using an image with a mask and without a mask in the dataset for each person, and the accuracy was 0.98.

It should also be noted that when adding images with a mask to the dataset, the system is more prone to false positives, as the neural network extracts fewer features and can easily be confused with another person who is also wearing a mask.

T-SNE has been used to display the 128 feature vectors in a 2-dimensional space. In the first plot 650 unmasked images of 25 people are used, and in the second plot 650 unmasked and 650 masked images of the same people are used.

<p align="center">
  <img src="https://user-images.githubusercontent.com/92673739/152246538-94dcde75-544a-4a41-8b25-a3fa137d7039.png" width="300"/>
  <img src="https://user-images.githubusercontent.com/92673739/152246723-00458161-f84e-474a-8968-bc91baadf3b3.png" width="300"/>
</p>

As can be seen, in the first graph, 25 groupings are made, corresponding to the 25 existing people, due to the similarity of their embeddings. On the other hand, in the second graph, 50 groupings are made, despite the fact that they are the same people. This shows that, by using a mask, the face is covered too much, and the embeddings are so different that the neural network thinks it is another person. Therefore, an unfamiliar person could be mistaken for a familiar person simply by wearing a mask.


## Speaker Recognition: *speaker_recognition.py*

Speaker recognition is implemented with *Librosa*, a Python package for audio and music analysis that provides the basic components needed to create auditory information retrieval systems. The features extracted from each audio are:
- MFCCs: coeﬁcients for ha-bla representation. They extract features from the components of an audio signal that are suitable for the identification of relevant content and ignore those with information that hinders the recognition process.
- Chroma diagram: a tool for categorising the pitch of voices. There are twelve different pitch classes.
- Mel-scaled spectrogram : the spectrogram is the result of the calculation of several spectra in superimposed ven-tana segments of the signal, through the Short-time Fourier Transform (stft). The frequencies of this spectrogram are converted to the mel scale, a perceptual musical scale of tones.
- Spectral contrast: the average energy in the upper quantile with that in the lower quantile of each subband into which the spectrogram is divided.
- Tonal centroid: is a representation that projects chromatic features in 6 dimensions. 

The extracted features are processed with *Sklearn* and will be the input to the pre-feed neural network, created with *Keras*. The SoftMax activation function is used to categorise the audios. The model is compiled with categorical cross-entropy as a loss function and the adam gradient descent algorithm as an optimiser.
For real-time audio retrieval, the *PyAudio* library is used, with which 32000 fps fragments are listened to and a prediction is made every seven seconds.

To run the programme, it is necessary to have created a model beforehand. To do this, you must have included your audio material in the folders 'train/' and 'validation/' in the folder 'Audios/'; all samples from the same person must be named the same, with a character at the end of the name to distinguish them. For the first run:

<!--sec data-collapse=true ces-->

    $ python speaker_recognition.py --train
    
<!--endsec-->

The program will create the neural network and save all the necessary models in the folder 'Models/Audios/' to be loaded in a future run. To run the program without retraining the model:

<!--sec data-collapse=true ces-->

    $ python speaker_recognition.py
    
<!--endsec-->

