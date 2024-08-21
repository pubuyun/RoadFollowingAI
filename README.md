# RoadFollowingAI

Using RobomasterS1(RC car) for collecting data, Road following AI based on Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN).

## Data Collection

For recording movements, you must first follow the instructions in the **s1_sdl_hack** folder to enable s1 for using sdk.
Then, run the Robomaster app, enter the control mode and run **record.py**, and you can see the video stream.
Control the car using the app while looking at the video stream, when you are ready, press **p** consistently to record data, and press **o** to end the program.

## Prepare Dataset

run **detection.py** to process images to lines.

## Training

run **main.py**, it will train automatically for 128 epochs and then test with a result [angle, speed]

\
referenced to Komanda's solution
