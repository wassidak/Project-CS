# 1. Import Dependencies
import cv2
import numpy as np
import os
import mediapipe as mp

# 2. Keypoints using MP Holistic
mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks,mp_holistic.FACEMESH_CONTOURS)  # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks,mp_holistic.POSE_CONNECTIONS)  # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS)  # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS)  # Draw right hand connections

def draw_styled_landmarks(image, results):
    # เปลี่ยนสีของจุด landmark
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                              )
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                              )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                              )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )

# 3. Extract Keypoint Values
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose,face,lh, rh])

# 4. Setup Folders for Collection
# Path for exported data, numpy arrays
DATA_PATH = os.path.join('Collect_Data')

# Actions that we try to detect
# array ของท่าทางภาษามือ
actions = np.array(['wait', 'สวัสดี', 'ขอบคุณ', 'ยินดีที่ได้รู้จัก', 'ขอโทษที่มาสาย',
                    'แล้วพบกันใหม่', 'โชคดี', 'คุณชื่ออะไร', 'คุณสบายดีไหม', 'ฉันสบายดี',
                    'ฉันไม่สบาย', 'ฉันไม่เป็นไร', 'วันนี้อากาศดี', 'คุณกินข้าวแล้วหรือยัง', 'ไปกินข้าวด้วยกันไหม',
                    'ฉันกำลังทานข้าว', 'พรุ่งนี้คุณจะไปไหน', 'ฉันกำลังจะกลับบ้าน', 'คุณจะกลับบ้านกี่โมง', 'คุณอายุเท่าไหร่',
                    'คุณเหนื่อยไหม', 'ฉันหิวข้าว', 'คุณทำงานอะไร', 'ฉันทำงานเป็นช่างทำผม', 'ฉันทำงานเป็นคนขับรถแท็กซี่',
                    'เมื่อวานไปไหนมา', 'เมื่อวานฉันไปเที่ยว', 'ฉันไปหาหมอ', 'ฉันขอโทษ', 'คุณเข้าใจไหม', 'ฉันไม่เข้าใจ'])

# 30 videos worth of data
# 30 folder = 30 frames
no_sequences = 30

# Videos are going to be 30 frames in length
# ความยาวของ 1 เฟรม เท่ากับ 30 วินาที
sequence_length = 30

# Folder start
start_folder = 30