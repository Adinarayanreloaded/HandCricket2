# HandCricket2
A project where the user plays hand cricket with computer using AI and ML.

Requirements

Python 3.7
Keras
Tensorflow
OpenCV

1.Clone the repository 

Run the command

git clone https://github.com/Adinarayanreloaded/HandCricket2.git

cd HandCricket2

2.Install the dependencies

pip install -r requirements.txt

(If it does not install directly you need to install it one by one all the dependencies in gitbash)

3.Gather Images by Using the ImagesP.py script for each gesture (one to six and none)

In the given example below we gather 70 images for the one gesture(Similarily it needs to be done for all the gestures)

python ImagesP.py one 70


4.We train the model using 

python TrainP.py

5.Test the model on some images whether it is predicting right or wrong 

python TestP.py <path_to_test_image>

6.Play the game with the computer 

python PlayP.py

Thank You 😊
