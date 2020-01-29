#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
import os
from PIL import Image
from PIL.ExifTags import TAGS
import folium 
import base64
import datetime
import webbrowser
import pyautogui
import glob


import cv2
import os
from keras.models import load_model
import numpy as np

#--------#

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.inference import load_image
from utils.preprocessor import preprocess_input

#

from operator import eq


def GPS_Marker(filename):
    extension = filename.split('.')[-1]
    if (extension == 'jpg') | (extension == 'JPG') | (extension == 'jpeg') | (extension == 'JPEG'):
        try:
            img = Image.open(filename)
            info = img._getexif()
            exif = {}
            for tag, value in info.items():
                decoded = TAGS.get(tag, tag)
                exif[decoded] = value
            # from the exif data, extract gps
            exifGPS = exif['GPSInfo']
            latData = exifGPS[2]
            lonData = exifGPS[4]
            # calculae the lat / long
            latDeg = latData[0][0] / float(latData[0][1])
            latMin = latData[1][0] / float(latData[1][1])
            latSec = latData[2][0] / float(latData[2][1])
            lonDeg = lonData[0][0] / float(lonData[0][1])
            lonMin = lonData[1][0] / float(lonData[1][1])
            lonSec = lonData[2][0] / float(lonData[2][1])
            # correct the lat/lon based on N/E/W/S
            Lat = (latDeg + (latMin + latSec / 60.0) / 60.0)
            if exifGPS[1] == 'S': Lat = Lat * -1
            Lon = (lonDeg + (lonMin + lonSec / 60.0) / 60.0)
            if exifGPS[3] == 'W': Lon = Lon * -1
            # print file
            msg = "There is GPS info in this picture located at " + str(Lat) + "," + str(Lon)
            print (msg)
            
        except:
                print ('There is no GPS info in this picture')
                pass
        name=filename.split('z')[0]
        txtname=filename.split('.')[0]
        f = open(txtname+ ".txt", 'r',encoding='UTF-8') 
        diary = f.read()

        
        pic = base64.b64encode(open(filename,'rb').read()).decode()
        image_tag = '<img src="data:image/jpeg;base64,{}"style="width:180px;height:200px;">'.format(pic)
        iframe = folium.IFrame(image_tag+diary, width=180, height=250)
        pop = folium.Popup(iframe, max_width=400)
        
        if name =='happy' : 
            #ic="star"
            col="pink"
        if name =='sad' : 
            col = "black"
        if name =='angry' : 
            #ic="star"
            col='red'
        if name =='neutral' : 
            #ic="cloud"
            col="green"
        if name =='surprise' :
            col="cadetblue"
        if name =='fear' :
            col ="blue"
        if name =='disgust' :
            col ='orange'
        
        cft=datetime.datetime.fromtimestamp(os.path.getctime(filename))
        strcft=cft.strftime('%Y-%m-%d %H:%M')
        print(strcft)
        toolt=name+" & "+strcft
       
        folium.Marker([str(Lat),str(Lon)],popup=pop,icon=folium.Icon(color=col),tooltip=toolt).add_to(m)
        m.save('testtest.html')

    
    
    
    
    
    



#카메라 실행 중에 아무 키나 입력 받으면 그 사진을 저장한다.
def start(folder_to_save):
    
    cv2.destroyAllWindows() # 재촬영 시 모두 포맷하고 사진 찍을 준비.
    
    while True:
        ret, frame = capture.read() # 실시간으로 프레임을 받아준다
        cv2.imshow("VideoFrame", frame)
        if cv2.waitKey(1) > 0: # 아무 키나 입력 할 때까지 실시간 카메라 보여줌
            #키 입력 시 사진 바로 captured_img에 저장됨
            
            cv2.imwrite(os.path.join(folder_to_save , 'captured_img.jpg'), frame) # 변수 활용(시간, 위치)
            return folder_to_save + '\captured_img.jpg'
            
def image_handler(folder_to_save):  # 사진 확인 및 재촬영 의사를 확인한다.
    #saved_path = ""
    while(True):
        saved_path = start(folder_to_save) # 아무 키나 눌렀을 때 저장되고, 그 사진의 저장된 경로를 반환 해준다.
        img_file = saved_path
        #img_file = r'C:\Users\kyung\Anaconda3\src\captured_img.jpg'  # 변수 활용(시간, 위치)
        img = cv2.imread(img_file,cv2.IMREAD_COLOR) 
        cv2.imshow('captured_image',img) 
    
        key = 12354312345
        key = cv2.waitKey(0) & 0xFF 
        if key == ord('r'): # r키 입력 시 재촬영
            #start(path)
            key = 1212312313
            continue
        elif key == ord('s'): # s 입력 시 저장
            #cv2.imwrite('copy_images/img1_copy.jpg',img) 
            return saved_path;
        #cv2.destroyAllWindow() # 권한이 없다고 한다.
        
def check_recoged_img(recoged_img_path): # recognition 함수 작동 후 저장된 얼굴 인식 결과 사진을 잠깐 확인 할 수 있게 해준다.
    img = cv2.imread(recoged_img_path,cv2.IMREAD_COLOR) 
    while(True):
        cv2.imshow('recoged_image',img) 

        if cv2.waitKey(1) > 0:
            break;


            
def recognition(f_2_s):
    tt = False  #이중 조건문을 탈출하기 위해서, goto문이 불가능하다 파이썬은..
    emotion_c = ""
    #recap == False
    # parameters for loading data and images
    #image_path = path_r
    image_path = image_handler(f_2_s)
    detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
    emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
    emotion_labels = get_labels('fer2013')
    font = cv2.FONT_HERSHEY_SIMPLEX

    # hyper-parameters for bounding boxes shape
    emotion_offsets = (20, 40)
    emotion_offsets = (0, 0)

    # loading models
    face_detection = load_detection_model(detection_model_path)
    emotion_classifier = load_model(emotion_model_path, compile=False)

    # getting input model shapes for inference
    emotion_target_size = emotion_classifier.input_shape[1:3]

    # loading images
    rgb_image = load_image(image_path, grayscale=False)
    gray_image = load_image(image_path, grayscale=True)
    gray_image = np.squeeze(gray_image)
    gray_image = gray_image.astype('uint8')

    faces = detect_faces(face_detection, gray_image)
    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]

        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            #pyautogui.confirm(text='one more', title='test', buttons=['ok', 'exit'])
            #recap = True
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
        emotion_text = ""
        emotion_text = emotion_labels[emotion_label_arg]
        
        
        
        #감정인식이 성공되면 감정 상태를 물어보고, 감정 확인 후 저장 or 탈출
        
        
        tof = pyautogui.confirm(text='Are you '+emotion_text+'?', title=emotion_text, buttons=['yes', 'no'])
        if(tof == 'yes'):
            tt = True
            
            emotion_c = emotion_text;
            
            color = (255, 0, 0) # 감정 정보 글씨 빨간색, 사각형도

            draw_bounding_box(face_coordinates, rgb_image, color)
            draw_text(face_coordinates, rgb_image, emotion_text, color, 0, -30, 1.5, 2)
        elif(tof == 'no'):
            tt = False
            break;
        
        #color = (255, 0, 0) # 감정 정보 글씨 빨간색, 사각형도

        #draw_bounding_box(face_coordinates, rgb_image, color)
        #draw_text(face_coordinates, rgb_image, emotion_text, color, 0, -30, 1.5, 2)
    if(tt == True):
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite('../src/'+ emotion_text +'.jpg', bgr_image) # 변수 활용
        check_recoged_img('../src/'+ emotion_text +'.jpg')
    else:
        pyautogui.alert(text='no emtion captured', title='error', button='OK')
    
    
    # 저장된 사진을 감정 예측 후 predicted_test_image.jpg로 저장
    
#def to_text(path):
    #return path+".txt"


#def refresh_map()
def Refreshmap() :
    path = r"C:\Users\kyung\Anaconda3\src"
    file_list = os.listdir(path)
    file_list_jpg= [file for file in file_list if file.endswith(".jpg")]
    for i in file_list_jpg:
        GPS_Marker(i)
        
    print ("file_list_jpg: {}".format(file_list_jpg))
    webbrowser.open('testtest.html')

   

            
###################################################################################################################################

if __name__ == '__main__': 
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 900)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    folder_to_save = r"C:\Users\kyung\Anaconda3\images"
    #global recap = False
    #recognition(image_handler(folder_to_save))
    
    recognition(folder_to_save)
    
    
    capture.release() 
    cv2.destroyAllWindows()
    
    
    m= folium.Map(location=[37.56,127],zoom_start=12.5)
    m.save(r"testtest.html")
    '''
    GPS_Marker(r"happyz0.jpg")
    GPS_Marker(r"happyz2.jpg")
    #GPS_Marker(r"Angryz.jpg")
    GPS_Marker(r"neutralz0.jpg")
    GPS_Marker(r"surprisez0.jpg")
    
    '''
    Refreshmap()
    #webbrowser.open(r"testtest.html")







#####################################################################################################

