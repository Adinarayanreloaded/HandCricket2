from keras.models import load_model
from cv2 import cv2
import numpy as np
import sys

filepath = sys.argv[1]

REV_CLASS_MAP = {
    0:"none",
    1:"one",
    2:"two",
    3:"three",
    4:"four",
    5:"five",
    6:"six"

}


def mapper(val):
    return REV_CLASS_MAP[val]


model = load_model("Hand-cricket2-model.h5")

# prepare the image
img = cv2.imread(filepath)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (227, 227))

# predict the move made
pred = model.predict(np.array([img]))
move_code = np.argmax(pred[0])
move_name = mapper(move_code)

print("Predicted: {}".format(move_name))
