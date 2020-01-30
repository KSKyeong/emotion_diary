#!/usr/bin/env python
# coding: utf-8

# In[5]:


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
from random import *



# !주석이 영어로 달린 부분은 인용한 함수들에 있던 주석! #









    
#####상경 코드################################################################################################################################################

#카메라 실행 중에 아무 키나 입력 받으면 그 사진을 저장한다.
def start(folder_to_save):
    
    cv2.destroyAllWindows() # 재촬영 시 모두 포맷하고 사진 찍을 준비.
    
    while True:
        ret, frame = capture.read() # 실시간으로 프레임을 받아준다
        cv2.imshow("VideoFrame", frame)
        if cv2.waitKey(1) > 0: # 아무 키나 입력 할 때까지 실시간 카메라 보여줌
            #키 입력 시 사진 바로 captured_img에 저장됨
            
            cv2.imwrite(os.path.join(folder_to_save , 'captured_img.jpg'), frame) # 처음 찍은 사진을 저장
            return folder_to_save + '\captured_img.jpg'  # 찍힌 사진의 경로를 반환해준다.
            
def image_handler(folder_to_save):  # 사진 확인 및 재촬영 의사를 확인한다.
    #saved_path = ""
    while(True):
        saved_path = start(folder_to_save) # 아무 키나 눌렀을 때 저장되고, 그 사진의 저장된 경로를 반환 해준다.
        img_file = saved_path # start()함수가 저장한 사진의 위치를 img_file에 저장,
        img = cv2.imread(img_file,cv2.IMREAD_COLOR) # 그 사진을 불러와서 보여줌
        cv2.imshow('captured_image',img) 
        
        # 분기를 나누어서 사진 저장 혹은 재촬영 한다.
        
        key = 12354312345
        key = cv2.waitKey(0) & 0xFF 
        if key == ord('r'): # r키 입력 시 재촬영, while문의 처음으로 돌아가서 다시 start() 함수를 호출한다.ㄴ
            #start(path)
            key = 1212312313
            continue
        elif key == ord('s'): # s 입력 시 저장
            #cv2.imwrite('copy_images/img1_copy.jpg',img) 
            return saved_path; # s 입력 시 while문 탈출 with 경로
        #cv2.destroyAllWindow() # 권한이 없다고 한다. 왜지 감자
        
def check_recoged_img(recoged_img_path): # recognition 함수 작동 후 저장된 얼굴 인식 결과 사진을 잠깐 확인 할 수 있게 해준다.
    img = cv2.imread(recoged_img_path,cv2.IMREAD_COLOR) 
    while(True):
        cv2.imshow('recoged_image',img) 

        if cv2.waitKey(1) > 0:
            break;
#아마 사용 안 할듯



            
def recognition(f_2_s):
    #이중 조건문을 탈출하기 위해서, goto문이 불가능하다 파이썬은..
    tt = False  # 이중 조건문 탈출 위한 변수 설정
    emotion_c = "" # 이중 조건문 탈출 위한 변수 설정
    
    
    #recap == False
    # parameters for loading data and images
    #image_path = path_r
    
    
    image_path = image_handler(f_2_s) # 저장 여부 확인까지 완료한 뒤에 저장된 사진의 경로를 반환,받음
    
    # 학습된 모델과 감정labels의 경로를 설정해준 부분
    detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
    emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
    emotion_labels = get_labels('fer2013')
    font = cv2.FONT_HERSHEY_SIMPLEX #폰트 --> emotion정보 보여줄 때 사용

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
        
        
        tof = pyautogui.confirm(text='Are you '+emotion_text+'?', title=emotion_text, buttons=['yes', 'no']) # 인식된 감정의 정답 여부 질문, 사용자의 입력을 받음
        if(tof == 'yes'): # 알림 창의 yes 버튼을 눌렀을 때
            tt = True # 이중 조건문 탈출 위해
            
            emotion_c = emotion_text;
            
            color = (255, 0, 0) # 감정 정보 글씨 빨간색, 사각형도
            
            #draw_bounding_box(face_coordinates, rgb_image, color)
            #draw_text(face_coordinates, rgb_image, emotion_text, color, 0, -30, 1.5, 2)
            
        elif(tof == 'no'): # 알림 창의 no 버튼을 눌렀을 때
            tt = False
            break;
        
        #color = (255, 0, 0) # 감정 정보 글씨 빨간색, 사각형도

        #draw_bounding_box(face_coordinates, rgb_image, color)
        #draw_text(face_coordinates, rgb_image, emotion_text, color, 0, -30, 1.5, 2)
        
    if(tt == True):
        # yes 버튼을 눌렀을 때는 그 감정 여부에 맞는 파일명을 지어서 사진 저장.
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        next_num = next_index(emotion_text) # 인식된 감정 상태와 같은 파일들이 몇개 있는지 정보 얻고 다음 숫자가 저장됨
        cv2.imwrite('../src/'+ emotion_text +'z'+ str(next_num) +'.jpg', bgr_image) #  새로운 감정 인식 사진이 생성된다.
        f = open(emotion_text +'z'+ str(next_num)+".txt", 'w', encoding="UTF8") # 그에 매칭되는 일기장도 생성.
        f.close()
        
        # 이후 인식 된 얼굴 범위와 감정 정보를 화면을 통해 사용자에게 보여줌
        img = cv2.imread(image_path,cv2.IMREAD_COLOR)
         
        draw_bounding_box(face_coordinates, img, color)
        draw_text(face_coordinates, img, emotion_text, color, 0, -30, 1.5, 2)
        
        while(True): # 키 입력을 기다리며 화면 정지
            cv2.imshow(image_path,img) 
            

            if cv2.waitKey(1) > 0:
                break;
        # 체크가 완료되면 함수 탈출.
        
        #check_recoged_img('../src/'+ emotion_text +'z'+ str(next_num) +'.jpg')    --이것은 얼굴에 사각형, 감정정보 입혀진 사진 저장
    
    
    else: # 알림 창을 띄워서 인식 된 감정이 없다는 것을 알려줌 -->> (인식의 오류 or 사용자가 생각한 감정과의 불일치 시)
        pyautogui.alert(text='no emtion captured', title='error', button='OK')
    
    
    
    
#def to_text(path):
    #return path+".txt"

# 디렉토리에 저장된 사진들을 모두 map에 저장하는 함수 (긴 시간 소요)
def Refreshmap() :
    path = r"C:\Users\kyung\Anaconda3\src"
    file_list = os.listdir(path)
    file_list_jpg = [file for file in file_list if file.endswith(".jpg")]
    for i in file_list_jpg:
        GPS_Marker(i)
        
    print ("file_list_jpg: {}".format(file_list_jpg))
    

    

# 감성 인식 된 사진 파일들의 감정 중, 같은 감정이 현재 몇개가 저장되어있는지 확인, 다음 저장 되어야 할 인덱스 값을 반환해줌.
def next_index(emotion):
    path = r"C:\Users\kyung\Anaconda3\src"
    file_list = os.listdir(path)
    file_list_jpg= [file for file in file_list if file.endswith(".jpg")]
    matching = [s for s in file_list_jpg if emotion in s]
    return len(matching)

# main 함수 --> 무한 반복문 내에서 실행 됨
def main():
    
    #초기 설정
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    folder_to_save = r"C:\Users\kyung\Anaconda3\images"
    #초기 설정
    
    recognition(folder_to_save) #핵심 함수 실행
    
    #m= folium.Map(location=[37.56,127],zoom_start=12.5)
    #m.save(r"testtest.html")
    '''
    key_2_use = 'one_more'
    while(True):
        
        key_2_use = pyautogui.confirm(text='?', title='!!', buttons=['quit', 'map' ,'refresh', 'one_more'])
        #key_2_use = cv2.waitKey(0) & 0xFF
        if key_2_use == 'quit': # 종료하고 싶다.
            break
        elif key_2_use == 'map': # s 입력 시 저장 
            map_open()
        elif key_2_use == 'refresh': # r키 입력 시 재촬영
            Refreshmap()
        elif key_2_use == 'one_more':
            main()
        else:
            break;
        '''
    
    capture.release() 
    cv2.destroyAllWindows()
    # 카메라 종료 및 창 닫기
    
    
def map_open():
    Refreshmap()
    webbrowser.open('testtest.html')
    
    
    
    
    
## 필규 코드 #################################################################################################################################
def GPS_Marker(filename): # 저장된 사진의 GPS 정보를 불러온 후, 지도 내에 GPS 값에 맞는 위치에 (감정정보 + 저장 날짜) 저장 함수.
    extension = filename.split('.')[-1]  #현재 디렉토리의 파일들의 이름의 .뒷부분을 추출 해낸다 (ex) happy.jpg --> "jpg"
    if (extension == 'jpg') | (extension == 'JPG') | (extension == 'jpeg') | (extension == 'JPEG'): # 사진 관련 파일만 선택
        try:  
            img = Image.open(filename)
            info = img._getexif()
            exif = {}
            for tag, value in info.items():
                decoded = TAGS.get(tag, tag)
                exif[decoded] = value
            # exif 데이터로부터 gps 정보를 추출한다.
            exifGPS = exif['GPSInfo']
            latData = exifGPS[2]
            lonData = exifGPS[4]
            # lat와 lon 값을 계산한다.
            latDeg = latData[0][0] / float(latData[0][1])
            latMin = latData[1][0] / float(latData[1][1])
            latSec = latData[2][0] / float(latData[2][1])
            lonDeg = lonData[0][0] / float(lonData[0][1])
            lonMin = lonData[1][0] / float(lonData[1][1])
            lonSec = lonData[2][0] / float(lonData[2][1])
            # N/E/W/S 값을 통해 lat,lon 값을 수정한다. 
            Lat = (latDeg + (latMin + latSec / 60.0) / 60.0)
            if exifGPS[1] == 'S': Lat = Lat * -1
            Lon = (lonDeg + (lonMin + lonSec / 60.0) / 60.0)
            if exifGPS[3] == 'W': Lon = Lon * -1
            msg = "There is GPS info in this picture located at " + str(Lat) + "," + str(Lon)
            print (msg)
            
        except:
            Lat = 37.597126 - (randrange(4)/10000)
            Lon = 126.955879 - (randrange(4)/10000)
            msg = "There is GPS info in this picture located at " + str(Lat) + "," + str(Lon)
            print (msg)
            
            
            pass
        name=filename.split('z')[0]  # 파일명을 z기준으로 앞 부분을 잘라서 name에 저장한다. (ex) happyz10.jpg --> "happy"
        txtname=filename.split('.')[0]  # 파일명을 .기준으로 앞 부분을 잘라서 txtname에 저장한다. (ex) happyz10.jpg --> "happyz10"
        f = open(txtname+ ".txt", 'r',encoding='UTF-8') # txtname과 같은 txt파일 하나를 생성해준다 (일기용)
        diary = f.read()

        
        pic = base64.b64encode(open(filename,'rb').read()).decode() # 사진 파일을 불러오는 작업
        image_tag = '<img src="data:image/jpeg;base64,{}"style="width:180px;height:200px;">'.format(pic)
        iframe = folium.IFrame(image_tag+diary, width=180, height=250) # 지도 파일 내에 아이콘 클릭시 보여줄 화면 생성
        pop = folium.Popup(iframe, max_width=400) # 팝업 창의 크기 설정
        
        # 감정의 이름 별 아이콘 색상을 결정해준다.
        
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
        
        cft=datetime.datetime.fromtimestamp(os.path.getctime(filename)) # 사진의 저장 시간을 불러온다.
        strcft=cft.strftime('%Y-%m-%d %H:%M') # 원하는 표현 서식 형태를 정해준다.
        print(strcft)
        toolt=name+" & "+strcft # 감정상태 & 날짜,시간 을 아이콘 tooltip에 전달할 변수에 저장.
       
        folium.Marker([str(Lat),str(Lon)],popup=pop,icon=folium.Icon(color=col),tooltip=toolt).add_to(m) # Marker함수를 사용하여 마커 생성
        m.save('testtest.html') # 추가한 사항이 있으니 지도를 새로 testtest.html에 저장

    
    

            
##########(실행 되는 부분)############################################################################################################

if __name__ == '__main__': 
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    folder_to_save = r"C:\Users\kyung\Anaconda3\images"
    
    m= folium.Map(location=[37.56,127],zoom_start=12.5)
    m.save(r"testtest.html")
    
    main()
    key_2_use = 'one_more' # 처음에는 main() 함수가 한번 실행되게 해줌
    while(True):
        
        # 사용자의 버튼 입력에 따라 종료 혹은 map 오픈 혹은 사진 재 촬영을 선택하게 해주는 분기
        key_2_use = pyautogui.confirm(text='click you want', title='wait!', buttons=['quit', 'map' ,'one_more'])
        #key_2_use = cv2.waitKey(0) & 0xFF
        if key_2_use == 'quit': # 종료하고 싶다.
            break
        elif key_2_use == 'map': # 'map'버튼 클릭 시 지도 띄움
            map_open()
        elif key_2_use == 'one_more': # 한번 더 사진을 찍고 저장하고싶다.
            main()
        else:
            break;







#####################################################################################################

