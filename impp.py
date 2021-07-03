import numpy as np
import base64
import cv2 
import keras 
import cv2
import json 
#import tensorflow as tf 




def load_stuff():
    global model
    model = keras.models.load_model('./model/model.h5')
    global class_name_to_number
    global _class_number_to_name
    with open ('./model/class_dictionary.json','r') as f:
        class_name_to_number = json.load(f)
    _class_number_to_name = {v:k for k,v in class_name_to_number.items()}

def get_cv2_image_from_base64_string(b64str):
    '''
    credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
    :param uri:
    :return:
    '''
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def get_cropped_image(base64 = None,file_path = None):
    if file_path:
        img = cv2.imread(file_path)
    else:
        img = get_cv2_image_from_base64_string(base64) 
    
    #configuring pretrained model for cat detection 
    config = './model/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    frozen_model = './model/frozen_inference_graph.pb'
    model = cv2.dnn_DetectionModel(frozen_model,config)
    global classLabels 
    classLabels = []
    
    file_name = './model/pretrained_labels.txt'
    
    with open(file_name,'rt') as fpt:
        classLabels = fpt.read().rstrip('\n').split('\n')
        classLabels.append(fpt.read())
    
    
    model.setInputSize(320,320)
    model.setInputScale(1.0/127.5)
    model.setInputMean((127.5,127.5,127.5))
    model.setInputSwapRB(True)


    cropped_faces = []

    #detecting any object present using pretrained model 
    ClassIndex , confidence , bbox = model.detect(img,confThreshold = 0.5)
    face_cascade = cv2.CascadeClassifier('./model/opencv/haarcascades/haarcascade_frontalcatface.xml')

    #box of object and cropping image if a cat face is present in it
    for x,y,w,h in bbox:
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        cropped_faces.append(roi_color)
        global face
        face = face_cascade.detectMultiScale(roi_gray)
        if len(face) != 0:
            return(cropped_faces)


def predict(imgs):
    for img in imgs:
        scaled_raw_image = cv2.resize(img,(32,32))
        scaled_raw_image = scaled_raw_image.reshape(-1,32,32,3)
        global model
        pred = model.predict(scaled_raw_image)
        value =   [np.argmax(element) for element in pred]
        return value

def class_number_to_name(class_num):
    return _class_number_to_name[class_num]

def send_predictions(b64):
    result = []
    imarray = get_cropped_image(base64 = b64)
    if imarray is not None:
        prediction = predict(imarray)
        #prediction = np.array(prediction)
        #prediction = prediction.reshape(-1)
        result.append(class_number_to_name(prediction[0]))
        return ('The cat looks like a ', result)
    else:
        return('Kindly submit a photo in which both eyes and the whole body of the cat is clearly visible')    


