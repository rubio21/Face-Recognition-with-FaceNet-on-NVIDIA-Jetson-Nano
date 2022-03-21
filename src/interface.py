import PySimpleGUI as sg
from PIL import Image
import cv2
import io
import os
import time
import face_recognition

sg.theme('Default1')

lista = [
    [sg.Text("Employees in:")],
    [sg.Listbox(values=[], size=(40, 20), key="-LIST-")],
    [sg.Image(filename="../logos/juntos.png")],
]

layout = [
    [
        [sg.Text('Face Recognition System', justification='center', font='Helvetica 20 bold')]
    ],
    [
        sg.Column([[sg.Image(key="-STREAM-")],]),
        sg.VSeparator(),
        sg.Column(lista, key='-HOME-', element_justification='center'),
        sg.Column([
            [sg.Text("Welcome to CVC", key="-WELCOME-", justification='center', font='Helvetica 20')],
            [sg.Image(key="-IMAGE-")],
            [sg.Image(filename="../logos/juntos.png")],
        ], key='-RECOGNITION-', visible=False, element_justification='center'),
    ],
    [
        sg.Button("Take a photo"),
        sg.Button("Exit")
    ]
]

fd = face_recognition.FaceDetection()
fr = face_recognition.FaceRecognition()
# Dataset embeddings
embeddings = fr.load_face_embeddings("../Dataset/", fd)
cap = cv2.VideoCapture(0)

c=0
window = sg.Window("Face Recognition", layout, resizable=True)
control=False
employees_list=[]

while True:
    event, values = window.read(timeout=0)
    if event == "Exit" or event == sg.WIN_CLOSED:
        break

    ret, frame = cap.read()
    imgbytes = cv2.imencode('.png', frame)[1].tobytes()
    window['-STREAM-'].update(data=imgbytes)
    frame = cv2.flip(frame, 1)
    reconocimiento = face_recognition.face_recognition(frame, embeddings, fd, fr, False)
    print(reconocimiento)
    if reconocimiento != []:
        image = Image.open("../Dataset/" + reconocimiento[0] + ".JPG")
        image.thumbnail((400, 400))
        bio = io.BytesIO()
        image.save(bio, format="PNG")
        window["-IMAGE-"].update(data=bio.getvalue())
        window["-WELCOME-"].update("Welcome " + reconocimiento[0])
        if reconocimiento[0] not in employees_list:
            employees_list.append(reconocimiento[0])
        window["-LIST-"].update(employees_list)
        window['-HOME-'].update(visible=False)
        window['-RECOGNITION-'].update(visible=True)
        c = 0
    else:
        c += 1
        if c == 25:
            window["-WELCOME-"].update("Welcome to CVC")
            window['-RECOGNITION-'].update(visible=False)
            window['-HOME-'].update(visible=True)

    if event == "Take a photo":
        persona = sg.popup_get_text(title='Save photo to database', message="Enter your name")
        if persona != None:
            cv2.imwrite("../Dataset/" + persona + "_1.jpg", frame)
            sg.popup('Image saved!', title='Save photo to database', image=sg.EMOJI_BASE64_HAPPY_THUMBS_UP)

window.close()