import cv2
faceClassif = cv2.CascadeClassifier("xml/haarcascade_frontalface_default.xml")
image = cv2.imread('lfw2/lfw2/Abel_Pacheco/Abel_Pacheco_0002.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = faceClassif.detectMultiScale(gray,
                                     minNeighbors=5,
                                     scaleFactor=1.1,
                                     minSize=(30,30),
                                     maxSize=(200,200))
for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()