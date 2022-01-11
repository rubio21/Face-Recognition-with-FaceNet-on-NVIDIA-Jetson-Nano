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


    
# Se guardan los audios en un dataframe de pandas con su hablante
def load_df(option):
	filelist = os.listdir('../audios/' + option)
	df = pd.DataFrame(filelist)
	df = df.rename(columns={0: 'file'})
	# print(df[df['file']=='.DS_Store'])
	speaker = []
	for i in range(0, len(df)):
		speaker.append(df['file'][i][:-5])
	df['speaker'] = speaker
	return df

# Función para extraer las propiedades de audio de cada archivo de audio
# Extrae 5 características numéricas de cada archivo: mfccs, chroma, mel, contraste y tonnetz
def extract_features(files, option):
	file_name = os.path.join(os.path.abspath('../audios/') + '/' + option + '/' + str(files.file))
	X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
	# Se generan coeficientes cepstrales de frecuencia Mel (MFCC) a partir de una serie temporal
	mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
	# Se genera una transformada de Fourier en tiempo corto (STFT) para utilizarla en chroma_stft
	stft = np.abs(librosa.stft(X))
	# Calcula un cromagrama a partir de una forma de onda o un espectrograma de potencia
	chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
	# Calcula un espectrograma en escala mel
	mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
	# Calcula el contraste espectral
	contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
	# Calcula las características del centroide tonal (tonnetz)
	tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
	
	return mfccs, chroma, mel, contrast, tonnetz


# Se concatenan todas las características numéricas de cada archivo para obtener una única matriz
# de 193 números para entrarla en la red neuronal.
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

def show_graphic(history):
	# Check out our train accuracy and validation accuracy over epochs.
	train_accuracy = history.history['accuracy']
	val_accuracy = history.history['val_accuracy']
	# Set figure size.
	plt.figure(figsize=(12, 8))
	# Generate line plot of training, testing loss over epochs.
	plt.plot(train_accuracy, label='Accuracy entrenamiento', color='#185fad')
	plt.plot(val_accuracy, label='Accuracy validación', color='orange')
	# Set title
	plt.title('Accuracy de entrenamiento y validación por épocas', fontsize=25)
	plt.xlabel('Época', fontsize=18)
	plt.ylabel('Validación cruzada categórica', fontsize=18)
	plt.xticks(range(0, 100, 5), range(0, 100, 5))
	plt.legend(fontsize=18);
	plt.show();

def prepare_data():
	train_df = load_df("train")
	validation_df = load_df("validation")
	test_df = load_df("test")

	# ----------------------------------------------------

	train_features = train_df.apply(extract_features, option='train', axis=1)
	X_train, y_train = concatenate_features(train_df, train_features)

	val_features = validation_df.apply(extract_features, option='validation', axis=1)
	X_val, y_val = concatenate_features(validation_df, val_features)

	test_features = test_df.apply(extract_features, option='test', axis=1)
	X_test, y_test = concatenate_features(test_df, test_features)

	# ------------------------------------------------------
	# Hot encoding y
	lb = LabelEncoder()
	train_unique = np.unique(y_train).tolist()
	with open('../Models/Audio/personas.txt','w') as outfile:
		json.dump(train_unique, outfile)
	y_train = to_categorical(lb.fit_transform(y_train))
	y_val = to_categorical(lb.fit_transform(y_val))
	print(y_train.shape, y_val.shape)

	ss = StandardScaler()
	X_train = ss.fit_transform(X_train)
	X_val = ss.transform(X_val)
	X_test = ss.transform(X_test)
	dump(ss, '../Models/Audio/SC_Model.bin', compress=True)
	return X_train, X_val, X_test, y_train, y_val, lb, test_df



def train_model(X_train, X_val, y_train, y_val):
	# Se construye un modelo denso simple con Early stopping y softmax para la clasificación categórica
	# El número de clases es la cantidad de hablantes diferentes que hay en el conjunto de train y validación
	model = Sequential()
	# La entrada son las 193 características de los audios y la primera capa oculta es de 193 nodos
	model.add(Dense(193, input_shape=(193,), activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(150, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(150, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(np.size(y_train[0]), activation='softmax'))
	model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
	early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')

	history = model.fit(X_train, y_train, batch_size=256, epochs=200,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stop])

	show_graphic(history)

	model.save('../Models/Audio/modelo.h5')

	return model


if __name__ == '__main__':
	X_train, X_val, X_test, y_train, y_val, lb, test_df = prepare_data()
	model = train_model(X_train, X_val, y_train, y_val)


	#model = load_model('../Models/Audio/modelo.h5')

	p = pyaudio.PyAudio()
	stream = p.open(format=pyaudio.paInt16, channels=2, rate=32000, input_device_index=11, frames_per_buffer = 32000, input=True, output=False)
	ss = load('../Models/Audio/SC_Model.bin')
	with open('../Models/Audio/personas.txt') as json_file:
		personas = json.load(json_file)
	c=0
	audio_frames=[]
	while(True):
		if(c>=7):
			waveFile = wave.open('../audios/realtime.wav', 'wb')
			waveFile.setnchannels(2)
			waveFile.setsampwidth(p.get_sample_size(pyaudio.paInt16))
			waveFile.setframerate(32000)
			waveFile.writeframes(b''.join(audio_frames))
			waveFile.close()
			X, sample_rate = librosa.load('../audios/realtime.wav', res_type='kaiser_fast')
			mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
			stft = np.abs(librosa.stft(X))
			chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
			mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
			contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
			tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
			features_array=[]
			features_array.append(np.concatenate((mfccs, chroma, mel, contrast, tonnetz)))
			pruebatest = np.array(features_array)
			pruebatest = ss.transform(pruebatest)
			predictions = model.predict(pruebatest)
			print("Valores: ", predictions)
			predicciones=np.argmax(predictions[0])
			print("PREDICCIONES: ", predicciones)
			confianza=np.max(predictions[0])
			print("CONFIANZA: ", confianza)
			confianza = (np.max(predictions[0])>=0.95)
			#predicciones=lb.inverse_transform(predicciones)
			print(np.where(confianza==True, personas[predicciones], 'Desconocido'))
			audio_frames=[]
			c=0
			time.sleep(2.0)
		else:
			print("Escuchando")
			data = stream.read(32000, exception_on_overflow=False)
			audio_frames.append(data)
			c+=1
	stream.stop_stream()
	stream.close()
	p.terminate()
	

	
