
from keras.models import load_model
from cv2 import cv2
import numpy as np
from random import choice
import os
import time

sum=0
REV_CLASS_MAP={
    0: "none",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six"
    
}


def mapper(val):
    return REV_CLASS_MAP[val]

def calculate(move1,move2):
   global sum,compsum
   if move1==move2:
    sum=-1
    return sum
   
   else:
        if move1==REV_CLASS_MAP.get(1):
            return 1
        elif move1==REV_CLASS_MAP.get(2):
            return 2
        elif move1==REV_CLASS_MAP.get(3):
            return 3
        elif move1==REV_CLASS_MAP.get(4):
            return 4
        elif move1==REV_CLASS_MAP.get(5):
            return 5
        else:
            return 6
def calculate_winner(sum1,sum2):
    if(sum1>sum2):
     return "User"
    elif(sum1<sum2):
     return "Computer"
    else: 
     return "Scores Tied!!"     
        
model = load_model("Hand-cricket2-model.h5")

cap = cv2.VideoCapture(0)
cap.set(3,1280)# 3-PROPERTY index for WIDTH
cap.set(4,720)# 4-PROPERTY index for HEIGHT 

prev_move = None

sum=0
compsum=0
turn='User'
winner='None'

while True:
    ret, frame = cap.read()
    if not ret:
        continue
        
    # rectangle for user to play
    cv2.rectangle(frame, (100, 100), (500, 500), (255, 255, 255), 2)
    # rectangle for computer to play
    cv2.rectangle(frame, (800, 100), (1200, 500), (255, 255, 255), 2)

    # extract the region of image within the user rectangle
    roi = frame[100:500, 100:500]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (227, 227))

    # predict the move made
    pred = model.predict(np.array([img]))
    move_code = np.argmax(pred[0])
    user_move_name = mapper(move_code)

    font=cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "First User turn then Comp", 
                (400,50), font, 0.7, (255,255,255),2, cv2.LINE_AA)
   
    # User Turn

    if prev_move != user_move_name:
        if user_move_name != "none":
            computer_move_name = choice(['one', 'two', 'three','four','five','six'])
            if turn=='User':
                if user_move_name!=computer_move_name:
                 sum=sum +calculate(user_move_name,computer_move_name)
                else:
                    print("out!!")
                    turn='Computer'
            else:
                if user_move_name!=computer_move_name:
                 compsum=compsum+ calculate(computer_move_name,user_move_name)
                else:
                    print("out!!")
                    turn='User'
                    winner=calculate_winner(sum,compsum)
                    sum=0
                    compsum=0
        else:
            computer_move_name = "none"
    prev_move = user_move_name
    
    # display the information
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Your Move: " + user_move_name,
                (50, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Computer's Move: " + computer_move_name,
                (750, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Sum: " + str(sum),
                (400, 600), font, 2, (0, 0, 255), 4, cv2.LINE_AA)
    cv2.putText(frame, "Sum: " + str(compsum),
                (500, 700), font, 2, (0, 0, 255), 4, cv2.LINE_AA)
    
    cv2.putText(frame, "Winner: " + winner,
                (400, 500), font, 2, (0, 0, 255), 4, cv2.LINE_AA)
    
    
    
    if computer_move_name != "none":
        icon = cv2.imread(
            "ImagesF2/{}.png".format(computer_move_name))    
        icon = cv2.resize(icon, (400,400))
        frame[100:500, 800:1200] = icon

    cv2.imshow("Hand Cricket Frame ", frame)

    k = cv2.waitKey(10)
    if k == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
    
    time.sleep(0.1)


