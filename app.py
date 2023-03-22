### 11. Test in Real Time
import cv2
import numpy as np
import mediapipe as mp
# import requests
# import configparser

from PIL import Image, ImageFont, ImageDraw
from flask import Flask, render_template, Response
from keras.models import model_from_json

from functions import *

app = Flask(__name__)

# เรียกใช้ model ที่เป็นไฟล์ .json
json_file = open("model.json", "r")
model_json = json_file.read()
# json_file.close()
model = model_from_json(model_json)
model.load_weights("action.h5")

colors = [(245,117,16), (117,245,16), (16,117,245), (0,0,0),(138,43,226),
            (0,100,0),(255,0,0), (240,128,128), (139,69,19),(105,105,105),
            (255,0,0), (240,128,128), (139,69,19),(105,105,105)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        # แถบสีที่ขึ้นบนข้อความแต่ละท่าทาง
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        # แถบข้อความด้านข้าง
        #cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        # show thai lang
        
        fontpath = "./BaiJamjuree-Regular.ttf" 

        font =  ImageFont.truetype(fontpath,30)
        output_pil = Image.fromarray(output_frame)
        draw = ImageDraw.Draw(output_pil)
        text = draw.text((3, 52+num*40), actions[num], font = font, fill=(255,255,255))

        output_frame = np.array(output_pil) 
        
    return output_frame

def generate():
    # 1. New detection variables
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.7

    #เปิดกล้องด้วย open cv
    cap = cv2.VideoCapture(0)

    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # while cap.isOpened():
        while True:

            # Read feed
            ret, frame = cap.read()
                
            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            print(results)
                        
            # Draw landmarks
            draw_styled_landmarks(image, results)
                        
            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.insert(0,keypoints)
            #sequence.append(keypoints)
            sequence = sequence[:40]
                    
                    
            if len(sequence) == 40:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))
                            
                            
            #3. Viz logic
                if np.unique(predictions[-40:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > threshold:   
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 3: 
                    sentence = sentence[-3:]

                    # Viz probabilities
                    #image = prob_viz(res, actions, image, colors)
                    
            #Output Text    
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1) 
            #cv2.putText(image, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # show thai lang
            fontpath = "./BaiJamjuree-Regular.ttf" 

            font =  ImageFont.truetype(fontpath,30) #truetype(font,ขนาดของ font)
            img_pil = Image.fromarray(image)
            draw = ImageDraw.Draw(img_pil)
            text = draw.text((3,0), ' '.join(sentence), font = font, fill=(255,255,255)) 
            #(3,0) คือระยะห่างของแกน x,y ของข้อความ / fill คือ ใส่สีตัวอักษรแบบ RGB 

            image = np.array(img_pil)
                    
            # Show to screen # ถ้าขึ้นเว็บไม่ต้องโชว์
            # cv2.imshow('OpenCV Feed', image)

            frame2 = cv2.imencode('.jpg', image)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')

            # Break gracefully
            # if cv2.waitKey(10) & 0xFF == ord('q'):
            #     break
            key = cv2.waitKey(20)
            if key == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__=="__main__":
    app.run(debug=True)
    # app.run(host="0.0.0.0", port=5000)
