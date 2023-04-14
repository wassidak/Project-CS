import numpy as np

from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, confusion_matrix, classification_report

from functions import *

json_file = open("model.json", "r")
model_json = json_file.read()
# json_file.close()
model = model_from_json(model_json)
model.load_weights("action.h5")

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

X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

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