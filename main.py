import cv2
import numpy as np
import face_recognition
import os
import pyttsx3

path = 'images'
images=[]

classNames = []
myList = os.listdir(path)
print(myList)

speaker = pyttsx3.init('sapi5')
voices= speaker.getProperty('voices')
speaker.setProperty('voice', voices[0].id)

for cl in myList:
    currentImg = cv2.imread(f'{path}/{cl}')
    images.append(currentImg)
    classNames.append(os.path.splitext(cl)[0])

print(classNames)

def findEncoadings(images):
    encodingList = []
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodingList.append(encode)
    return encodingList

encodeListKnown = findEncoadings(images)
print('Encoading Done')

cap = cv2.VideoCapture(0)

while True:
    success,img = cap.read()
    imgs= cv2.resize(img,(0,0),None,0.25,0.25)
    imgs = cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)


    facesCurrFrame = face_recognition.face_locations(imgs)
    encodesCurrFrame = face_recognition.face_encodings(imgs,facesCurrFrame)

    for encodeFace,faceLoc in zip(encodesCurrFrame,facesCurrFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis=face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            speaker.say("You Are " + name)
            speaker.runAndWait()
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)

