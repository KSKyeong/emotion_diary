#!/usr/bin/env python
# coding: utf-8

# In[4]:


import sys

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

#####################################################################################################


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
        img_file = r'C:\Users\kyung\Anaconda3\images\captured_img.jpg'  # 변수 활용(시간, 위치)
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
        cv2.imshow('captured_image',img) 

        if cv2.waitKey(1) > 0:
            break;


            
def recognition(f_2_s):
    
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
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
        emotion_text = ""
        emotion_text = emotion_labels[emotion_label_arg]

        color = (255, 0, 0) # 감정 정보 글씨 빨간색, 사각형도

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_text, color, 0, -30, 1.5, 2)
    #if(emotion_text == ""):
        #recognition(f_2_s)
        
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('../images/predicted_test_image.png', bgr_image) # 변수 활용
    check_recoged_img('../images/predicted_test_image.png')
    # 저장된 사진을 감정 예측 후 predicted_test_image.jpg로 저장
   

            
###################################################################################################################################

if __name__ == '__main__': 
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 900)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    folder_to_save = r"C:\Users\kyung\Anaconda3\images"
    
    #recognition(image_handler(folder_to_save))
    recognition(folder_to_save)
    
    
    capture.release() 
    cv2.destroyAllWindows()







#####################################################################################################


# In[ ]:




