import os
import librosa
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
import pyaudio
import wave
import time
from joblib import dump, load
import json
import argparse
import cv2
import Jetson.GPIO as GPIO


led_pin = [7,11]

def initialise_led():
	GPIO.setmode(GPIO.BOARD)
	GPIO.setup(led_pin, GPIO.OUT)
	# Red LED on and green LED off
	GPIO.output(led_pin, (GPIO.HIGH, GPIO.LOW))

def change_led(green_state):
	if (green_state):
		# Green LED on
		GPIO.output(led_pin, (GPIO.HIGH, GPIO.HIGH))
	else:
		# Green LED off
		GPIO.output(led_pin, (GPIO.HIGH, GPIO.LOW))

# Audios are stored in a pandas dataframe with their speaker.
def load_df(option):
	filelist = os.listdir('../audios/' + option)
	df = pd.DataFrame(filelist)
	df = df.rename(columns={0: 'file'})
	speaker = []
	for i in range(0, len(df)):
		speaker.append(df['file'][i][:-5])
	df['speaker'] = speaker
	return df

# Function to extract the audio properties of each audio file.
# Extracts 5 numeric characteristics from each file: mfccs, chroma, mel, contrast and tonnetz.
def extract_features(files, option=None, training=False):
	if training:
		file_name = os.path.join(os.path.abspath('../audios/') + '/' + option + '/' + str(files.file))
		X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
	else:
		X, sample_rate = librosa.load(files, res_type='kaiser_fast')
	mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
	stft = np.abs(librosa.stft(X))
	chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
	mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
	contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
	tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
	
	return mfccs, chroma, mel, contrast, tonnetz


# All numerical features from each file are concatenated to obtain a single 193-number matrix for input into the neural network.
def concatenate_features(df, features):
	features_array = []
	for i in range(0, len(features)):
		features_array.append(np.concatenate((
		features[i][0],
		features[i][1],
		features[i][2],
		features[i][3],
		features[i][4]), axis=0))
	X_ = np.array(features_array)
	y_ = np.array(df['speaker'])
	return X_, y_

# Accuracy graph of training and validation is shown.
def show_graphic(history):
	train_accuracy = history.history['accuracy']
	val_accuracy = history.history['val_accuracy']
	plt.figure(figsize=(12, 8))
	plt.plot(train_accuracy, label='Training Accuracy', color='#185fad')
	plt.plot(val_accuracy, label='Validation Accuracy', color='orange')
	plt.title('Training and Validation Accuracy by Epoch', fontsize=25)
	plt.xlabel('Epoch', fontsize=18)
	plt.ylabel('Categorical Crossentropy', fontsize=18)
	plt.xticks(range(0, 100, 5), range(0, 100, 5))
	plt.legend(fontsize=18);
	plt.show();

# Prepare the training and validation sets that will be used to build the neural network.
def prepare_data():
	train_df = load_df("train")
	validation_df = load_df("validation")

	train_features = train_df.apply(extract_features, option='train', training=True, axis=1)
	X_train, y_train = concatenate_features(train_df, train_features)

	val_features = validation_df.apply(extract_features, option='validation', training=True, axis=1)
	X_val, y_val = concatenate_features(validation_df, val_features)


	lb = LabelEncoder()
	train_unique = np.unique(y_train).tolist()
	with open('../Models/Audio/persons.txt','w') as outfile:
		json.dump(train_unique, outfile)
	y_train = to_categorical(lb.fit_transform(y_train))
	y_val = to_categorical(lb.fit_transform(y_val))
	ss = StandardScaler()
	X_train = ss.fit_transform(X_train)
	X_val = ss.transform(X_val)
	dump(ss, '../Models/Audio/SC_Model.bin', compress=True)
	return X_train, X_val, y_train, y_val, lb



def train_model(X_train, X_val, y_train, y_val):
	# A simple dense model with Early stopping and softmax is built for categorical classification.
	# The number of classes is the number of different speakers in the training and validation set.
	model = Sequential()
	# The input is the 193 characteristics of the audios and the first hidden layer is 193 nodes.
	model.add(Dense(193, input_shape=(193,), activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(np.size(y_train[0]), activation='softmax'))
	model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
	early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')

	history = model.fit(X_train, y_train, batch_size=256, epochs=200,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stop])
	show_graphic(history)
	model.save('../Models/Audio/speaker_recognition_model.h5')
	return model

# Main function
def real_time(model):
	# PyAudio is used to record audio in real time.
	# The parameters entered are related to the characteristics of the camera, and vary according to the environment in which it is used. 
	p = pyaudio.PyAudio()
	stream = p.open(format=pyaudio.paInt16, channels=2, rate=32000, input_device_index=11, frames_per_buffer = 32000, input=True, output=False)
	ss = load('../Models/Audio/SC_Model.bin')
	with open('../Models/Audio/persons.txt') as json_file:
		persons = json.load(json_file)
	c=0
	audio_frames=[]
	initialise_led()
	while(True):
		if(c>=7):
			#The previous 7 seconds are loaded, stored in 'real_time.wav'.
			waveFile = wave.open('../Models/Audio/real_time.wav', 'wb')
			waveFile.setnchannels(2)
			waveFile.setsampwidth(p.get_sample_size(pyaudio.paInt16))
			waveFile.setframerate(32000)
			waveFile.writeframes(b''.join(audio_frames))
			waveFile.close()
			# Feature extraction
			mfccs, chroma, mel, contrast, tonnetz = extract_features('../Models/Audio/real_time.wav')
			features_array=[]
			# Concatenation and tranformation with Standard Scaler
			features_array.append(np.concatenate((mfccs, chroma, mel, contrast, tonnetz)))
			test = ss.transform(np.array(features_array))
			predictions = model.predict(test)
			print("Values: ", predictions)
			predicciones=np.argmax(predictions[0])
			print("Predictions: ", predicciones)
			confidence=np.max(predictions[0])
			print("Confidence: ", confidence)
			confidence = (np.max(predictions[0])>=0.95)
			# Prediction only if confidence is bigger than 95%. 
			print(np.where(confidence==True, persons[predicciones], 'Unknown'))
			if confidence>=0.95: change_led(True)
			audio_frames=[]
			c=0
			time.sleep(2.0)
			change_led(False)
		else:
			# 32000 frames per second recording every 7 seconds
			print("Listening...")
			data = stream.read(32000, exception_on_overflow=False)
			audio_frames.append(data)
			c+=1


		
	stream.stop_stream()
	stream.close()
	p.terminate()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--train", action="store_true")
	args = parser.parse_args()
	if(args.train):
		X_train, X_val, y_train, y_val, lb = prepare_data()
		model = train_model(X_train, X_val, y_train, y_val)
	else:
		model = load_model('../Models/Audio/speaker_recognition_model.h5')
	real_time(model)
