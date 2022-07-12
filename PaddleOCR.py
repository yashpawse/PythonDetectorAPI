from paddleocr import PaddleOCR
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from flask import Flask,request,json
import json
import cv2
from urllib.request import urlopen

app = Flask(__name__);

@app.route('/api/checkImage', methods=['POST'])
def checkIdentity():
    request_Json = request.json #convert response object to JSON
    imageLink = request_Json["imageLink"]
    fName = request_Json["fName"]
    lName = request_Json["lName"]
    idNo = request_Json["idNo"]

    detectedTexts=[];
    response = requests.get(imageLink)
    resp = urlopen(imageLink)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
    cv2.imshow('image', image)
    result = ocr.ocr(image, cls=True)
    for item in result:
        if(item[1][1]>=0.80): #Confidence
            detectedTexts.append(str(item[1][0]).replace(" ","").lower()) #Text

    containsDetailsObject = {
        "isfNameExists": any(str(fName).replace(" ","").lower() in item for item in detectedTexts),
        "islNameExists": any(str(lName).replace(" ","").lower() in item for item in detectedTexts),
        "isidNoExists": any(str(idNo).replace(" ","").lower() in item for item in detectedTexts),
        "additionalInfo": detectedTexts
    }

    jsonObject = json.dumps(containsDetailsObject)
    return jsonObject;

@app.route('/api/checkFace', methods=['GET'])
def checkProfilePhoto():
    face_cascade = cv2.CascadeClassifier('face_detector.xml')
    img = cv2.imread('result.jpg')
    faces = face_cascade.detectMultiScale(img, 1.1, 4)
    for (x, y, w, h) in faces: 
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imwrite("face_detected.jpg", img) 

if __name__ == "__main__":
    app.run(port=8080,host='0.0.0.0')