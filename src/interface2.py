import PySimpleGUI as sg
from PIL import Image
import cv2
import io
import os
import time
import face_recognition


logos = [
    [sg.Image(filename="../logos/cvc.png")],
    [sg.Image(filename="../logos/uab.png")],
]

boton=[
    [sg.Button("Start system")],
]
lista = [
    [sg.Text("Employees in:")],
    [sg.Listbox(values=["Employee 1", "Employee 2", "Employee 3", "Employee 4", "Employee 5"], size=(40, 20), key="-LIST-")],
]
prueba= [
    [sg.Image(key="-IMAGE-")],
]
layout = [
    [
        [sg.Text('Face Recognition System', key="-TITLE-", justification='center', font='Helvetica 20', background_color='#EE9444')]
    ],
    [
        sg.Column(logos),
        sg.VSeparator(),
        sg.Column(boton, key='-BUTTON-'),
        sg.Column(lista, key='-HOME-', visible=False),
        sg.Column(prueba, key='-RECOGNITION-', visible=False),
    ],
    [
        [sg.Text("Welcome to CVC", key="-WELCOME-", justification='center', font='Helvetica 20', background_color='#EE9444')],
    ]
]

fd = face_recognition.FaceDetection()
fr = face_recognition.FaceRecognition()
# Dataset embeddings
embeddings = fr.load_face_embeddings("../Dataset/", fd)
cap = cv2.VideoCapture(0)

c=0
window = sg.Window("Face Recognition", layout, resizable=True, background_color='#B0B0B0')
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    if event == "Start system":
        window['-BUTTON-'].update(visible=False)
        window['-HOME-'].update(visible=True)

        while True:
            ret, image = cap.read()
            image = cv2.flip(image, 1)
            reconocimiento = face_recognition.face_recognition(image, embeddings, fd, fr, False)
            print(reconocimiento)
            if reconocimiento != []:
                image = Image.open("../Dataset/" + reconocimiento[0] + ".JPG")
                image.thumbnail((400, 400))
                bio = io.BytesIO()
                image.save(bio, format="PNG")
                window["-IMAGE-"].update(data=bio.getvalue())
                window["-WELCOME-"].update("Welcome " + reconocimiento[0])
                window['-HOME-'].update(visible=False)
                window['-RECOGNITION-'].update(visible=True)
                c = 0
            else:
                c += 1
                if c == 25:
                    window["-WELCOME-"].update("Welcome to CVC")
                    window['-RECOGNITION-'].update(visible=False)
                    window['-HOME-'].update(visible=True)
            window.refresh()
window.close()
