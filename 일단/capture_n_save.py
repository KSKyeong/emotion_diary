#!/usr/bin/env python
# coding: utf-8

# In[4]:


import cv2
import os
import numpy as np

def image_handler():  
    while(True):
        img_file = r'C:\Users\kyung\Anaconda3\images\captured_img.jpg' 
        img = cv2.imread(img_file,cv2.IMREAD_COLOR) 
        cv2.imshow('captured_image',img) 
    
        key = 12354312345
        key = cv2.waitKey(0) & 0xFF 
        if key == ord('r'): # r키 입력 시 재촬영
            start()
            key = 1212312313
            continue
        elif key == ord('s'): # s 입력 시 저장
            #cv2.imwrite('copy_images/img1_copy.jpg',img) 
            break;
    #cv2.destroyAllWindow() # 권한이 없다고 한다.



##########################################@@@@@@@@@@@@@@@@@@@@@@@@@@@

#카메라 실행 중에 아무 키나 입력 받으면 그 사진을 저장한다.
def start():
    
    cv2.destroyAllWindows() # 재촬영 시 모두 포맷하고 사진 찍을 준비.
    
    while True:
        ret, frame = capture.read() # 실시간으로 프레임을 받아준다
        cv2.imshow("VideoFrame", frame)
        if cv2.waitKey(1) > 0: # 아무 키나 입력 할 때까지 실시간 카메라 보여줌
            #키 입력 시 사진 바로 captured_img에 저장됨
            path = r"C:\Users\kyung\Anaconda3\images" 
            cv2.imwrite(os.path.join(path , 'captured_img.jpg'), frame) 
        
            
            
            # 그 찍은 사진을 불러와서 확인 시켜주고 다시 닫음
            
            #img_file = r'C:\Users\kyung\Anaconda3\images\captured_img.jpg' 
            #img = cv2.imread(img_file,cv2.IMREAD_COLOR)
            #cv2.imshow('title',img) 
            #cv2.waitKey(0) & 0xFF 
            
            break;
            


if __name__ == '__main__': 
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    
    start() #촬영을 시작한다.
    image_handler() # 사진 확인 및 재촬영 의사를 확인한다.
    
    
    
    capture.release() 
    cv2.destroyAllWindows()

