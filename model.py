import os
import numpy as np

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, confusion_matrix, classification_report
from functions import *

### 6. Preprocess Data and Create Labels and Features
label_map = {label:num for num, label in enumerate(actions)}
print(label_map)

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

# print(np.array(sequences).shape)
# print(np.array(sequences).size)
# print(np.array(sequences).ndim)
# print(np.array(labels).shape)

X = np.array(sequences)
# print(X.shape)
y = to_categorical(labels).astype(int)

# แบ่งข้อมูล train 80% : test 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print()
print("แบ่งข้อมูล: \n", X_train.shape, X_test.shape, y_train.shape, y_test.shape, "\n")


### 7. Build and Train LSTM Neural Network
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Sequential api
model = Sequential()

model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))

model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

model.add(Dense(actions.shape[0], activation='softmax'))

#64,128,32 คือจำนวนโหนดของโมเดล
#input_shape=(30,258) >> 30 คือเฟรมรูป 258 คือจำนวน keypoint (จุดบนมือ+หน้า)


#จำนวนคำ/ประโยคที่เราเทรนเข้าไป
print("จำนวนประโยคทั้งหมด = ", actions.shape[0])
res = [0.7, 0.2,0.1]
print(action[np.argmax(res)], "\n")

# optimizer='Adam' คืออัลกอริทึมที่มาช่วยเพิ่มประสิทธิในการทำงานของโมเดล
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train model
# epochs=800 คือจำนวนรอบในการเทรนโมเดล
# เทรนเพื่อนให้ค่า loss ลดลง (น้อยกว่า 0) และให้ค่า accuracy เพิ่มขึ้น (เข้าใกล้ 1)
# สร้างโฟลเดอร์ Logs ขึ้นมา
# ไม่ต้องเทรนทุกรอบก็ได้
model.fit(X_train, y_train, epochs=500, batch_size=200, callbacks=[tb_callback])
print()
model.summary()

# เขียนไฟล์ .json
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json) 


### 8. Make Predictions
res = model.predict(X_test)
print()
print(actions[np.argmax(res[0])])
print(actions[np.argmax(y_test[0])])

### 9. Save Weights
model.save('action.h5')
model.load_weights('action.h5')

### 10. Evaluation using Confusion Matrix and Accuracy
yhat = model.predict(X_test)

ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

print()
m_confus = multilabel_confusion_matrix(ytrue, yhat)
print("multilabel confusion matrix : \n", m_confus, "\n")

confus = confusion_matrix(ytrue, yhat)
print("confusion matrix : \n", confus, "\n")

print("classification report : \n", classification_report(ytrue, yhat, digits=4), "\n")

#ความแม่นยำของข้อมูลที่ test ไป
print("accuracy score =", accuracy_score(ytrue, yhat), "\n")

#ความแม่นยำของชุดข้อมูลการทดสอบ
(ls,acc)=model.evaluate(x=X_test,y=y_test)
print('TEST MODEL ACCURACY = {}%'.format(acc*100))

#ความแม่นยำของชุดข้อมูลการฝึก
(ls,acc)=model.evaluate(x=X_train,y=y_train)
print('TRAIN MODEL ACCURACY = {}%'.format(acc*100))