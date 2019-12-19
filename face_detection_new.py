import cv2
import os
import matplotlib.pyplot as plt

num_of_sample = 10
vid = cv2.VideoCapture(0)#to open the camera
face_cascade = cv2.CascadeClassifier('C:\\Users\KAVITA.LAPTOP-E2HD8FAQ\Anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
file_name = input('Enter the name of person : ')
dir_path='C://Users/KAVITA.LAPTOP-E2HD8FAQ/Contacts/Desktop/orl_face/'+str(file_name)
os.mkdir(dir_path)
a=os.path.abspath(dir_path)

iter1=0
while(iter1<num_of_sample):
    r,frame = vid.read()
    frame = cv2.resize(frame,(640,480))
    im1=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)#GRAY SCALE CONVESION OF COLOR IMAGE
    face=face_cascade.detectMultiScale(im1)
    for x,y,w,h in (face):
        cv2.rectangle(frame,(x,y),(x+w,y+h),[255,0,0],4)
        iter1=iter1+1
        im_f = im1[y:y+h,x:x+w]
        im_f = cv2.resize(im_f,(112,92))
        cv2.putText(frame,'Face No. '+ str(iter1),(x,y),cv2.FONT_ITALIC,1,(255,0,255),2,cv2.LINE_AA)
        path2 = str(a) + str('\\%d.png')%(iter1)
        cv2.imwrite(path2,im_f)
    cv2.imshow('frame',frame)
    cv2.waitKey(1)
vid.release()
cv2.destroyAllWindows()

