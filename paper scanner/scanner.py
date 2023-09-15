# Import essential libraries
import requests
import cv2
import numpy as np
import imutils
import os
import cv2
import imutils
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import requests
import time
from base64 import b64encode
from IPython.display import Image
from pylab import rcParams

# Replace the below URL with your own. Make sure to add "/shot.jpg" at last.
url = "http://192.168.0.137:8080//shot.jpg"
img_counter = 0
working_directory = os.getcwd()
        
# While loop to continuously fetching data from the Url
while True:
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    img = imutils.resize(img, width=1000, height=1800)
    cv2.imshow("Android_cam", img)
  
    # Press Esc key to exit
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(os.path.join(working_directory , 'paper scanner/' + img_name), img)
        print("{} written!".format(img_name))
        img_counter += 1
  
cv2.destroyAllWindows()

rcParams['figure.figsize'] = 10, 20

def makeImageData(imgpath):
    img_req = None
    with open(imgpath, 'rb') as f:
        ctxt = b64encode(f.read()).decode()
        img_req = {
            'image': {
                'content': ctxt
            },
            'features': [{
                'type': 'DOCUMENT_TEXT_DETECTION',
                'maxResults': 1
            }]
        }
    return json.dumps({"requests": img_req}).encode()

def requestOCR(url, api_key, imgpath):
  imgdata = makeImageData(imgpath)
  response = requests.post(ENDPOINT_URL, 
                           data = imgdata, 
                           params = {'key': api_key}, 
                           headers = {'Content-Type': 'application/json'})
  return response

with open(os.path.join(working_directory , 'paper scanner/vision_api.json')) as f:
    data = json.load(f)

ENDPOINT_URL = 'https://vision.googleapis.com/v1/images:annotate'
api_key = data["api_key"]
img_loc = os.path.join(working_directory , 'paper scanner/' + img_name)

Image(img_loc)

result = requestOCR(ENDPOINT_URL, api_key, img_loc)

if result.status_code != 200 or result.json().get('error'):
    print ("Error")
else:
    result = result.json()['responses'][0]['textAnnotations']

newfile = open(os.path.join(working_directory , 'paper scanner/handwritten_essay.txt'), "w+")

for index in range(len(result)):
    print(result[index]["description"])
    newfile.write(result[index]["description"])
newfile.close()