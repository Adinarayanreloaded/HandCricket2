from keras.models import load_model
import cv2
import numpy as np
from random import choice

REV_CLASS_MAP = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "none":7
}


def mapper(val):
    return CLASS_MAP[val]


def calculate_winner(move1, move2, sum):
    if move1 == move2:
        out = 810498
        return out

    if move1 == "one":
        if move2 == "two":
            sum = sum + 3
            return sum
        if move2 == "three":
            sum = sum + 4
            return sum
        if move2 == "four":
            sum = sum + 5
            return sum
        if move2 == "five":
            sum = sum + 6
            return sum
        if move2 == "six":
            sum = sum + 7
            return sum

    if move1 == "two":
        if move2 == "one":
            sum = sum + 3
            return sum
        if move2 == "three":
            sum = sum + 5
            return sum
        if move2 == "four":
            sum = sum + 6
            return sum
        if move2 == "five":
            sum = sum + 7
            return sum
        if move2 == "six":
            sum = sum + 8
            return sum

    if move1 == "three":
        if move2 == "one":
            sum = sum + 4
            return sum
        if move2 == "two":
            sum = sum + 5
            return sum
        if move2 == "four":
            sum = sum + 7
            return sum
        if move2 == "five":
            sum = sum + 8
            return sum
        if move2 == "six":
            sum = sum + 9
            return sum

     if move1 == "four":
        if move2 == "one":
            sum = sum + 5
            return sum
        if move2 == "two":
            sum = sum + 6
            return sum
        if move2 == "three":
            sum = sum + 7
            return sum
        if move2 == "five":
            sum = sum + 9
            return sum
        if move2 == "six":
            sum = sum + 10
            return sum
        
     if move1 == "five":
        if move2 == "one":
            sum = sum + 6
            return sum
        if move2 == "two":
            sum = sum + 7
            return sum
        if move2 == "three":
            sum = sum + 8
            return sum
        if move2 == "four":
            sum = sum + 9
            return sum
        if move2 == "six":
            sum = sum + 11
            return sum
        
     if move1 == "six":
        if move2 == "one":
            sum = sum + 7
            return sum
        if move2 == "two":
            sum = sum + 8
            return sum
        if move2 == "three":
            sum = sum + 9
            return sum
        if move2 == "four":
            sum = sum + 10
            return sum
        if move2 == "five":
            sum = sum + 11
            return sum
        
        
model = load_model("Hand-cricket2-model.h5")

cap = cv2.VideoCapture(0)

prev_move = None

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

    # predict the winner (human vs computer)
    if prev_move != user_move_name:
        sum = 0
        if user_move_name != "none":
            computer_move_name = choice(['one', 'two', 'three','four','fivr','six'])
            winner = sum
            winner = calculate_winner(user_move_name, computer_move_name,sum)
            k = winner
            if k == 810498:
            break
        else:
            computer_move_name = "none"
            winner = 0
    prev_move = user_move_name

    # display the information
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Your Move: " + user_move_name,
                (50, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Computer's Move: " + computer_move_name,
                (750, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Winner: " + winner,
                (400, 600), font, 2, (0, 0, 255), 4, cv2.LINE_AA)

    if computer_move_name != "none":
        icon = cv2.imread(
            "images/{}.png".format(computer_move_name))
        icon = cv2.resize(icon, (400, 400))
        frame[100:500, 800:1200] = icon

    cv2.imshow("Rock Paper Scissors", frame)


cap.release()
cv2.destroyAllWindows()
