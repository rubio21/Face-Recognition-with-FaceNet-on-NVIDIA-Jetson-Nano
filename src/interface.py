import PySimpleGUI as sg
from PIL import Image
import cv2
import io
import face_recognition
from time import sleep
from threading import Thread, currentThread
import os
import unidecode
import copy

dataset = '../Dataset/'
sg.theme('Default1')




def count_down(window):
    t = currentThread()
    count = 5
    while True:
        if not getattr(t, "do_run", True):
            count=5
        window.write_event_value("COUNTDOWN", count)
        count -= 1
        sleep(1)

lista = [
    [sg.Text("Employees in:")],
    [sg.Listbox(values=[], size=(30, 20), key="-LIST-")],
]

layout = [
    [
        [sg.Text('Face Recognition System', justification='center', font='Helvetica 20 bold')]
    ],
    [
        sg.Column([[sg.Image(key="-STREAM-")],]),
        sg.VSeparator(),
        sg.Column(lista, key='-HOME-'),
        sg.Column([
            [sg.Text("Welcome to CVC", key="-WELCOME-", justification='center', font='Helvetica 20')],
            [sg.Image(key="-IMAGE-")],
        ], key='-RECOGNITION-', visible=False, element_justification='center'),
    ],
    [
        sg.Text('  ', key='COUNTDOWN', font='Helvetica 20 bold', visible=False),
    ],
    [
        sg.Button("Take a photo"),
        sg.Button("Report error"),
        sg.Button("Contribute an improvement"),
        sg.Button("Exit")
    ],
    [
        [sg.Image(filename="../logos.png")],
    ]
]

fd = face_recognition.FaceDetection()
fr = face_recognition.FaceRecognition()
# Dataset embeddings
fr.load_face_embeddings(dataset, fd)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # set new dimensionns to cam object (not cap)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

window = sg.Window("Face Recognition",layout, element_justification='c').Finalize()
window.Maximize()

control=False
employees_list=[]
list_count=0
tp=0
verification_array = []
t = Thread(target=count_down, args=(window,), daemon=True)
t.start()
t.do_run = False

while True:
    event, values = window.read(timeout=0)
    if event == "Exit" or event == sg.WIN_CLOSED:
        break

    ret, frame = cap.read()
    height, width, _ = frame.shape
    imgbytes = cv2.imencode('.png', cv2.resize(frame,(700, 394)))[1].tobytes()

    if tp == 1:
        faces = fd.detect_faces(frame)
        print(len(faces))
        if len(faces) == 0:
            t.do_run = False
        elif len(faces) == 1:
            x, y, w, h = faces[0]
            area = w+h
            if area < 600:
                t.do_run = False
                if 0 < (y+h) < height and 0 < (x+w) < width:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                imgbytes = cv2.imencode('.png', cv2.resize(frame, (700, 394)))[1].tobytes()
            else:
                t.do_run = True
                frame2 = copy.deepcopy(frame)
                if 0 < (y+h) < height and 0 < (x+w) < width:
                    cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)
                imgbytes = cv2.imencode('.png', cv2.resize(frame2, (700, 394)))[1].tobytes()
        else:
            t.do_run = False
            for face in faces:
                x, y, w, h = face
                if 0 < (y+h) < height and 0 < (x+w) < width:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                imgbytes = cv2.imencode('.png', cv2.resize(frame, (700, 394)))[1].tobytes()

    window['-STREAM-'].update(data=imgbytes)
    frame = cv2.flip(frame, 1)
    reconocimiento = face_recognition.face_recognition(frame, fd, fr, False)
    print(reconocimiento, verification_array)
    if tp == 0:
        if reconocimiento != []:
            if len(verification_array)<2:
                verification_array.append(reconocimiento[0])
            else:
                if all(j == verification_array[0] for j in verification_array):
                    image = Image.open(dataset + reconocimiento[0] + ".JPG")
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
                    list_count = 0
                verification_array=[]
        else:
            list_count += 1
            if list_count >= 25:
                window["-WELCOME-"].update("Welcome to CVC")
                window['-RECOGNITION-'].update(visible=False)
                window['-HOME-'].update(visible=True)

    if event == "Take a photo":
        tp = 1
        sg.popup('If you are wearing glasses, please take them off', title='Save photo to database',no_titlebar=True)
        window['COUNTDOWN'].update(visible=True)
        window['-HOME-'].update(visible=False)
        window['-RECOGNITION-'].update(visible=False)

    if event == "Report error":
        text = sg.popup_get_text('Type your error')
        if text is not None:
            check_files = 0
            while True:
                if os.path.isfile('../errors/'+str(check_files)+'.txt'):
                    check_files+=1
                else:
                    break
            with open('../errors/'+str(check_files)+'.txt', 'w') as f:
                f.write(text)
            sg.popup_timed('Thank you for your help!', title='Report error', image=sg.EMOJI_BASE64_HAPPY_THUMBS_UP, no_titlebar=True)

    if event == "Contribute an improvement":

        text = sg.popup_get_text('Type your idea')
        if text is not None:
            check_files = 0
            while True:
                if os.path.isfile('../improvements/'+str(check_files)+'.txt'):
                    check_files += 1
                else:
                    break
            with open('../improvements/'+str(check_files)+'.txt', 'w') as f:
                f.write(text)
            sg.popup_timed('Thank you for your help!', title='Contribute an improvement', image=sg.EMOJI_BASE64_HAPPY_THUMBS_UP, no_titlebar=True)

    if 'COUNTDOWN' in values:
        window['COUNTDOWN'].update(value=values['COUNTDOWN'])
        if values['COUNTDOWN'] == 0:
            t.do_run = False
            window['COUNTDOWN'].update(visible=False)
            faces = fd.detect_faces(frame)
            like = sg.popup("Do you like it?", title='Save photo to database', image=cv2.imencode('.png', cv2.resize(frame,(700, 394)))[1].tobytes(), button_type=1)
            if like == 'Yes':
                persona = sg.popup_get_text(title='Save photo to database', message="Enter your name",no_titlebar=True)
                persona = unidecode.unidecode(persona)
                if persona != None:
                    save_frame = copy.deepcopy(frame)
                    x, y, w, h = faces[0]
                    if (y - int(h / 2) > 0) and (x - w > 0) and (y + h + int(h / 2) < height) and (x + w * 2 < width):
                        save_frame = save_frame[y - int(h / 2):y + h + int(h / 2), x - w:x + w * 2]
                    cv2.imwrite(dataset + persona + ".JPG", save_frame)
                    fr.new_embedding(frame, faces, persona)
                    sg.popup_timed('Image saved!', title='Save photo to database',image=sg.EMOJI_BASE64_HAPPY_THUMBS_UP, no_titlebar=True)

            tp = 0

window.close()