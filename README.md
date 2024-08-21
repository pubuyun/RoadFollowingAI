# RoadFollowingAI

Using RobomasterS(rc car) for collecting data, Road following ai based on Convolutional Neural Networks (CNN) and Recurrent Neural Network(RNN).

## Data Collection

For recording movements, you must first follow the instructions in **s1_sdl_hack** folder to enable s1 for using sdk.
Then, run the Robomaster app, enter the control mode and run **record.py**, and you can see the video stream.
Control the car using app while looking at the video stream, when you are ready, press **p** consistantly to record data, press **o** to end the program.

## Prepare Dataset

run **detection.py** to process images to lines.

## Training

run **main.py**, it will train automaticly for 128 epoch then test with a result [angle, speed]

\
referenced to komanda's solution
