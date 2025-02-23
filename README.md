# Open-CV-python-code-for-trained-modules-and-data-set-craeter
import cv2
import numpy as np


faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam=cv2.VideoCapture(0);

id = input('Enter User ID: ')   #id variable

sampleNum=0
Max_sample = 50

while(True):
	ret,img=cam.read();
	#print(img)
	#print(ret)
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	cv2.imshow("gray",gray);
	
	faces=faceDetect.detectMultiScale(gray,1.3,5);
	for(x,y,w,h) in faces:
		sampleNum=sampleNum+1;
		cv2.imwrite("dataSet/User."+str(id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])#whoever faces detect it will write that face for that use a id variable
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
		cv2.waitKey(100);
	cv2.imshow("Face",img);
	cv2.waitKey(1);
	if(sampleNum>Max_sample):
		break
cam.release()
cv2.destroyAllWindows()
// Above code is for Data set creater 



import os, sys
import cv2
import numpy as np
import pickle
import random
import time
from datetime import datetime
import mail
import glob
from mail import*
from PIL import Image
import shutil
# from openpyxl import load_workbook
import pandas as pd
import serial 
import smtplib 
from email.mime.multipart import MIMEMultipart 
from email.mime.text import MIMEText 
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email import encoders

capture_duration = 60
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam=cv2.VideoCapture(0)
rec=cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer//model.yml")

id=0
count = 0
count_ = 0
ret,img=cam.read()
start_time = time.time()
ids = '$,'

def report_send_mail(label, image_path):
    with open(image_path, 'rb') as f:
        img_data = f.read()
    fromaddr = "daminmain@gmail.com"
    toaddr = "dhinaexposure@gmail.com"
    msg = MIMEMultipart() 
    msg['From'] = fromaddr 
    msg['To'] = toaddr 
    msg['Subject'] = "Alert"
    body = str(label)
    msg.attach(MIMEText(body, 'plain'))  # attach plain text
    image = MIMEImage(img_data, name=os.path.basename(image_path))
    msg.attach(image) # attach image
    s = smtplib.SMTP('smtp.gmail.com', 587) 
    s.starttls() 
    s.login(fromaddr, "kpqtxqskedcykwjz") 
    text = msg.as_string() 
    s.sendmail(fromaddr, toaddr, text) 
    s.quit()
        
while( int(time.time() - start_time) < capture_duration):

    _,img=cam.read()
    font = cv2.FONT_HERSHEY_PLAIN
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.rectangle(img,(x-50,y-50),(x+w+50,y+h+50),(255,0,0),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        if (conf<60):
            if id == 1 or id == 2 or id == 3:
                cv2.imwrite('image.jpg', img)
                image_path = 'image.jpg'
                label_ = id
                report_send_mail(label_, image_path)
        else:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255, 0, 0),2)
            cv2.putText(img,'Unknown Person', (x,y+400),font,2,(255, 0, 0),2)
            
    cv2.imshow("Face",img)
    if(cv2.waitKey(1)==ord('q')):
        break

cam.release()
cv2.destroyAllWindows()

// this above code  is for Face detection 


import os
import cv2
import numpy as np
from PIL import Image


recognizer=cv2.face.LBPHFaceRecognizer_create();


path='dataSet/'



def getImagesWithID(path):     #loop through all the images for that use a functions
	imagePaths=[os.path.join(path,f) for f in os.listdir(path)]  #to use a list .join to append the path with images 
	
	
	faces=[]
	IDs=[]
	for imagePath in imagePaths:
		faceImg=Image.open(imagePath).convert('L');
		faceNp=np.array(faceImg,'uint8')
		ID=int(os.path.split(imagePath)[-1].split('.')[1])
		faces.append(faceNp)
		print( ID )
		IDs.append(ID)
		cv2.imshow("training",faceNp)
		cv2.waitKey(10)
	return IDs,faces


Ids,faces=getImagesWithID(path)
recognizer.train(faces,np.array(Ids))
recognizer.write('recognizer/model.yml')
cv2.destroyAllWindows()
// this code is for training the cv for the face recogination
