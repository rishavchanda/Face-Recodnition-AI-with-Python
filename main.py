import cv2
import numpy as np
import face_recognition

imgRishav = face_recognition.load_image_file('images/rishav2.jpg')
imgRishav=cv2.cvtColor(imgRishav,cv2.COLOR_BGR2RGB)

imgRishavTest = face_recognition.load_image_file('images/rishav2.jpg')
imgRishavTest=cv2.cvtColor(imgRishavTest,cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imgRishav)[0]
encodeElon = face_recognition.face_encodings(imgRishav)[0]
cv2.rectangle(imgRishav,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(0,255,0),2)


facelocTest = face_recognition.face_locations(imgRishavTest)[0]
encodeElonTest = face_recognition.face_encodings(imgRishavTest)[0]
cv2.rectangle(imgRishavTest,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(0,255,0),2)


cv2.imshow('Rishav Chanda',imgRishav)
cv2.imshow('Rishav Test',imgRishavTest)
cv2.waitKey(0)
