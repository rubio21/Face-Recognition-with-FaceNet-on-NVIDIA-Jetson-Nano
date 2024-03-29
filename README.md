# Face Recognition with FaceNet on NVIDIA Jetson Nano
This repository documents a face recognition system implemented on an NVIDIA Jetson Nano. An interface has been designed, with the aim of integrating the system into a television to display live recognitions. The system does not perform any further functionality beyond recognising people, but it could be integrated into an entry and exit system such as an automatic door, as the NVIDIA Jetson Nano is capable of performing these actions.

<h4> Extras that are not integrated into the interface: </h4> 

To complement the recognition system, a speaker recognition algorithm is developed. The Jetson Nano is used to be able to send outputs to devices when recognising a person. In my case, I use the Logitech HD Pro C920 camera to get the images and sounds and two LEDs to simulate an output when a recognition is obtained.


<h2> Table of contents </h2>

<p>
    <img src="https://user-images.githubusercontent.com/92673739/152345376-8a5bf484-c6e1-4eec-87f3-931892c8a8c9.jpg" width="300" align="right"/>
    <ul>
        <li> <a href=#mp> Main program: interface and face recognition </li>
        <ul>
            <li> <i> interface.py </i> </a> </li>
        </ul>
        <li> <a href=#fr> Face Recognition </a></li>
        <ul>
            <li> <a href=#fr1> <i> first_tests.py </i> </a> </li>
            <li> <a href=#fr2> <i> detect_faces.py </i> </a> </li>
            <li> <a href=#fr3> <i> face_recognition.py </i> </a> </li>
        </ul>
        <li> <a href=#sr> Speaker Recognition </li>
        <ul>
            <li> <i> speaker_recognition.py </i> </a> </li>
        </ul>
        <li> <a href=#co> Considerations </a> </li>
        <ul>
            <li> <a href=#co1> Face mask in face recognition with FaceNet </a> </li>
            <li> <a href=#co2> Program to capture video and audio </a> </li>
        </ul>
        </ul>
</p>

<h2 id="mp"> Main program: interface and face recognition </h2>
The main functionality of the project can be found in the <i> interface.py </i> and <a href=#fr3> <i> face_recognition.py </i> </a> files, the rest of the files in the repository are the result of tests and implementations that have been carried out during the project until the development of the final program.

By executing the <i> interface.py </i> file, all the necessary models for face detection and recognition will be loaded and the 'Dataset/' directory will be processed to generate the feature vectors of all the people that the system must know. The default appearance of the interface is as follows:
<br /> <br />

<p align="center"> <img src="https://user-images.githubusercontent.com/92673739/181227266-771d178d-8e93-4cac-9c6b-0b1e38e35843.png" width="700"/> </p>
<br /> <br />

The interface displays the live content captured by the camera and a list of all people who have been recognised. To add more interaction and feedback with the users, the interface has 4 buttons:
- 'Take a photo': To expand the dataset, the functionality to add a new person during the run has been developed. The program recommends removing glasses to avoid future confusion, makes sure that there is no more than one face at the moment of taking the photo and checks that the person is close enough to the camera. It counts down from 5 seconds to take the picture, provided all conditions are met.
- 'Report error': Users can report an error they have observed, which will be saved in a '.txt' in the 'errors/' folder.
- 'Contribute an improvement': Users can report an idea they have come up with to improve the program, which will be saved in a '.txt' file in the 'improvements/' folder.
- 'Exit': Clicking on this button will terminate the execution.

<h3> Recognition of a person </h3>
As soon as a person stands in front of the camera and is recognised by the system, the interface will change its appearance: <br /> <br />

<p align="center"> <img src="https://user-images.githubusercontent.com/92673739/181227262-6e391e38-95ac-4c14-b345-62eb097708b7.png" width="700"/> </p>
<br /> <br />

The list of recognised persons will disappear to show the image of the recognised person and a welcome message. When this person has left and a few seconds have passed, the interface will return to its initial state and redisplay the list, including this person.


<h2 id="fr"> Face Recognition </h2>
The experiments are divided into face detection and face recognition algorithms. Both are started with older techniques (Viola-Jones and HOG in face detection and Eigenfaces, Fisherfaces and LBPH in face recognition). After obtaining and comparing the results, Single-Shot Detector and FaceNet are used for the final program.

<h3 id="fr1"> <i> first_tests.py </i></h3>
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

The OpenCV library has been used in the experiments carried out. A model has been created for each of the three face recognition procedures described above, based on previously detected faces. For the detected face, a prediction of the most similar being from the training set is generated, associated with a confidence factor. This factor is different for each procedure, so a threshold has been used to classify a prediction as "unknown" in case it is exceeded, being 4500 in Eigenfaces, 450 in Fisherfaces and 70 in LBP.
The three options coincide in frequent recognition errors in images with an environment and illumination unequal to those of the training set. In addition, numerous samples of each person are needed to train the patterns, so it takes too much memory and the model takes too long to create. Therefore, these techniques do not provide the necessary reliability and speed.


<h3 id="fr2"> <i> detect_faces.py </i></h3>

This file detects faces in an image or video from a Deep Neural Network. The network is used through OpenCV, which includes a DNN module that allows pre-trained neural networks to be loaded; this greatly improves speed, reduces the need for dependencies and most models are very light in size. The neural network used is based on **Single-Shot Detector (SSD)** with a ResNet base network.
Single-Shot Detector (SSD) is an algorithm that detects the object (face) in a single pass over the input image, unlike other models that traverse it more than once. SSD is based on the use of convolutional networks that produce multiple bounding boxes of various fixed sizes and score the presence of the object in those boxes, followed by a suppression step to produce the final detections.

In the code implementation, the pre-trained model is read through two files and the test images are fed into the network, which returns the detections found. The files are:
- *opencv_face_detector_uint8.pb*: contains the model.
- *opencv_face_detector.pbtxt*: contains the configuration for the model.

The process makes use of a BLOB (Binary Large Object), which is an element used to store dynamically changing large data; in this case, the *dnn.BlobFromImage* function handles the preprocessing, which includes setting the dimensions of the blob (used as input for the image) and normalisation.
Then, the blob is passed through the network with *net.setInput(blob)* and detections and predictions are obtained with *net.forward()*. For greater precision, the detections obtained are run through and their associated confidence value is compared with a threshold, in order to eliminate the least reliable ones.

The results obtained have been very good, since an algorithm that greatly improves Viola Jones and HOG is achieved, showing itself to be accurate and robust in the detection of faces with different viewing angles and lighting and occlusion conditions.

<p align="center"> <img src="https://user-images.githubusercontent.com/92673739/151672189-fd33fb34-d2ff-4eb2-a6e8-aa611f2a95ef.jpg" width="500"/> </p>

In this example, an accuracy of 100% is obtained. Unlike the previous methods, this algorithm has been able to detect both faces with glasses and the face from the side of the camera.

It is convenient to use this file when you want to test only the detector, or when you want to save a new person in the dataset. The program captures the best confidence image provided by SSD from a video or stream and stores it in the faces folder.

<h3 id="fr3"> <i> face_recognition.py </i></h3>
This file contains the code from which face detection and face recognition in real time are performed.

Previously used methods for face recognition involve the need for large data for a single person and a training time for each new addition to the dataset. To avoid this drawback we use **One-Shot Learning**, a computer vision technique that allows the learning of information about faces from a single image, and without the need to re-train the model.
Face recognition in the project is done by **FaceNet**, a system that uses a deep convolutional network. The network is pre-trained through a triple loss function, which encourages vectors of the same person to become more similar (smaller distance) and those of different individuals to become less similar (larger distance).
The generalised operation of the system consists of transforming each face in the dataset into a 128-feature vector, which is called embedding. For each entry, the same transformation is applied to the detected faces and their identity with the most similar embedding in the dataset is predicted (as long as the difference is not greater than a verification threshold).

Results of running face_recognition.py with *Labeled Faces in the Wild* images. The left image of each pair is the one located in the dataset and the one on the right is the test image:
<p align="center"> <img src="https://user-images.githubusercontent.com/92673739/152239732-aec9413d-c308-4de9-8f94-b73ffade57c4.png" width="500"/> </p>


The images of the persons should be included in the folder 'Dataset/' (it is important that no more than one person appears in the same photo). The files containing the pre-trained neural networks are located in the 'FaceDetection/' and 'FaceRecognition/' folders of 'Models/'.

To run the system showing the recognitions:
<!--sec data-collapse=true ces-->

    $ python face_recognition.py --show
    
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
  - *new_embedding()*: is used when a new person is added to the database from the interface.
- *face_recognition()*: Use the two classes above to find the faces in an image and recognise them, comparing their embedding with all those in the dictionary.
- *main_program()*: Use *face_recognition()* to get recognitions throughout the video (or image).


<h2 id="sr"> Speaker Recognition: </h2>

Speaker recognition is implemented with *Librosa*, a Python package for audio and music analysis that provides the basic components needed to create auditory information retrieval systems. The features extracted from each audio are:
- MFCCs: coeﬁcients for speech representation. They extract features from the components of an audio signal that are suitable for the identification of relevant content and ignore those with information that hinders the recognition process.
- Chroma diagram: a tool for categorising the pitch of voices. There are twelve different pitch classes.
- Mel-scaled spectrogram: the spectrogram is the result of the calculation of several spectra in superimposed window segments of the signal, through the Short-time Fourier Transform (stft). The frequencies of this spectrogram are converted to the mel scale, a perceptual musical scale of tones.
- Spectral contrast: the average energy in the upper quantile with that in the lower quantile of each subband into which the spectrogram is divided.
- Tonal centroid features: is a representation that projects chromatic features in 6 dimensions. 

The extracted features are processed with *Sklearn* and will be the input to the feed forward neural network, created with *Keras*. The SoftMax activation function is used to categorise the audios. The model is compiled with categorical cross-entropy as a loss function and the adam gradient descent algorithm as an optimiser.
For real-time audio retrieval, the *PyAudio* library is used, with which 32000 fps fragments are listened to and a prediction is made every 7 seconds.

To run the program, it is necessary to have created a model beforehand. To do this, you must have included your audio material in the folders 'train/' and 'validation/' in the folder 'Audios_dataset/'; all samples from the same person must be named the same, with a character at the end of the name to distinguish them. You can add as many audio files to the training and validation sets as you wish. The quantity and quality of the files you include will determine the accuracy of the system. For the first run:

<!--sec data-collapse=true ces-->

    $ python speaker_recognition.py --train
    
<!--endsec-->

The program will create the neural network and save all the necessary models in the folder 'Models/Audios/' to be loaded in a future run. To run the program without retraining the model:

<!--sec data-collapse=true ces-->

    $ python speaker_recognition.py
    
<!--endsec-->

Each time a new person is added to the audio dataset, the model will have to be retrained.


<h2 id="co"> Considerations: </h2>
<h3 id="co1"> Face mask in face recognition with FaceNet </h3>

In view of the current COVID-19 situation, it is essential to wear a mask. In the case of wanting to implement the system, it must be considered that it is not able to recognise a person wearing a mask if their image in the dataset is with their face uncovered.

In order to be able to use it even with masks, the image must be added to the dataset. An experiment has been carried out using an image with a mask and without a mask in the dataset for each person, and the accuracy was 0.98.

It should also be noted that when adding images with a mask to the dataset, the system is more prone to false positives, as the neural network extracts fewer features and can easily be confused with another person who is also wearing a mask.

T-SNE has been used to display the 128 feature vectors in a 2-dimensional space. In the first plot 650 unmasked images of 25 people are used, and in the second plot 650 unmasked and 650 masked images of the same people are used.

<p align="center">
  <img src="https://user-images.githubusercontent.com/92673739/152246538-94dcde75-544a-4a41-8b25-a3fa137d7039.png" width="300"/>
  <img src="https://user-images.githubusercontent.com/92673739/152246723-00458161-f84e-474a-8968-bc91baadf3b3.png" width="300"/>
</p>

As can be seen, in the first graph, 25 groupings are made, corresponding to the 25 existing people, due to the similarity of their embeddings. On the other hand, in the second graph, 50 groupings are made, despite the fact that they are the same people. This shows that, by using a mask, the face is covered too much, and the embeddings are so different that the neural network thinks it is another person. Therefore, an unfamiliar person could be mistaken for a familiar person simply by wearing a mask.

<h3 id="co2"> Program to capture video and audio </h3>

In the 'src/' folder, the *record_video_audio.py* file is also provided, with which you can save audio, video, or both at the same time. The program contains the classes AudioRecorder() and VideoRecorder() for recording. *PyAudio* library is used for audio and *Cv2* for video.

To record audio:
<!--sec data-collapse=true ces-->
    $ python record_video_audio.py --record_audio --time duration_in_seconds    
<!--endsec-->

To record video:
<!--sec data-collapse=true ces-->
    $ python record_video_audio.py --record_video --time duration_in_seconds
<!--endsec-->

To record video and audio:
<!--sec data-collapse=true ces-->
    $ python record_video_audio.py --record_video_and_audio --time duration_in_seconds
<!--endsec-->

The data entered for the audio input are those corresponding to the camera used in this project, which has an integrated microphone. If you use another device, it is likely to give you errors and you will have to change the parameters.
