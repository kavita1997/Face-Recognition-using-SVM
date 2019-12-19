
# coding: utf-8

# In[27]:


import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mimg
from sklearn import svm
import sklearn
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from sklearn.externals import joblib

temp_path='C:\\Users\KAVITA.LAPTOP-E2HD8FAQ\Contacts\Desktop\orl_face' 
d_list = list() 

try: 
    for f in os.listdir(temp_path): 
        if os.path.isdir(os.path.join(temp_path, f)):
            d_list.append(f)
            
except: 
    print("\nError, once check the path") 
d=len(d_list)
print(d)


samp=7
train_data=np.zeros((7*d,8748))
train_label=np.zeros((7*d))
count=-1
plt.figure(1)
plt.ion()
for i in range(1,d+1):
    for j in range(1,samp+1):
        #plt.cla()
        count=count+1
        #print(count)
        path ='C:/Users/KAVITA.LAPTOP-E2HD8FAQ/Contacts/Desktop/orl_face/u%d/%d.png'%(i,j)
        im=mimg.imread(path)
        feat=hog(im)
        train_data[count,:]=feat
        train_label[count]=i
test_data=np.zeros(((10-samp)*d,8748))
test_label=np.zeros((10-samp)*d)
count=-1
for i in range(1,d+1):
    for j in range(samp+1,11):
        #plt.cla()
        count=count+1
        path ='C:/Users/KAVITA.LAPTOP-E2HD8FAQ/Contacts/Desktop/orl_face/u%d/%d.png'%(i,j)
        im=mimg.imread(path)
        feat=hog(im)
        test_data[count,:]=feat
        test_label[count]=i

classifier = svm.SVC(kernel='linear',C=1,gamma='auto')# Linear Kernel

classifier.fit(train_data,train_label)
y_pred=classifier.predict(test_data)
c = confusion_matrix(test_label, y_pred)
print(c)
a1=classifier.score(test_data,test_label)
print('Accuracy with linear ',a1)
joblib.dump(classifier,'images.pkl')
#print(classification_report(test_label, y_pred))

classifier1 = svm.SVC(kernel='rbf',C=1,gamma='auto')# Linear Kernel
classifier1.fit(train_data,train_label)
y_pred1=classifier1.predict(test_data)
c1 = confusion_matrix(test_label, y_pred1)
print(c1)
a2=classifier1.score(test_data,test_label)
print('Accuracy with rbf ',a2)

classifier2 = svm.SVC(kernel='poly',C=1,gamma='auto')# Linear Kernel
classifier2.fit(train_data,train_label)
y_pred2=classifier2.predict(test_data)
c2 = confusion_matrix(test_label, y_pred2)
print(c2)
a3=classifier2.score(test_data,test_label)
print('Accuracy with poly ',a3)
plt.title('Accuracy graph')
plt.xlabel('data')
plt.ylabel('target')
a=['linear','rbf','poly']
b=[a1,a2,a3]
plt.bar(a,b,color=['red','green','blue'],width=0.40)

#plt.legend()
plt.show()

