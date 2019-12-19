import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.externals import joblib
model=joblib.load('images.pkl')

vid = cv2.VideoCapture(0)#to open the camera
face_cascade = cv2.CascadeClassifier('C:\\Users\KAVITA.LAPTOP-E2HD8FAQ\Anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
iter1=0
while True:
    r,frame = vid.read()
    frame = cv2.resize(frame,(640,480))
    im1=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)#GRAY SCALE CONVESION OF COLOR IMAGE
    face=face_cascade.detectMultiScale(im1)
    for x,y,w,h in (face):
        cv2.rectangle(frame,(x,y),(x+w,y+h),[255,0,0],4)
        iter1=iter1+1
        im_f = im1[y:y+h,x:x+w]
        im_f = cv2.resize(im_f,(112,92))
        feat=hog(im_f)
        name=model.predict(feat.reshape(1,-1))
        print(name)       
        cv2.putText(frame,'Face No. '+ str(name),(x,y),cv2.FONT_ITALIC,1,(255,0,255),2,cv2.LINE_AA)
    cv2.imshow("frame", frame)
    #out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q') : # exit on q
        break   
vid.release()
cv2.destroyAllWindows()


