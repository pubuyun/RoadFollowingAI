import cv2
import time
import pandas as pd
from pynput import keyboard
import numpy as np
from matplotlib import pyplot as plt
import robomaster
from robomaster import robot
from robomaster import camera
from .utils.rcutil import *

v1 = np.arange(0, 1, 0.00001)
t1 = (1/6.6)*(-v1-np.log(1-v1))
plt.plot(t1,v1)

t2 = np.arange(0, 1.6, 0.0001)
v2 = -0.6*t2 + 1
plt.plot(t2,v2)

ac = [0,0.76,0.88,0.94,0.96,1,1]

de = list()
for i in range(0,17):
   de.append(round(-0.6*i/10 + 1,2))
de.append(0), de.append(0)

ep_robot = robot.Robot()
# 路由器组网
ep_robot.initialize(conn_type='sta')
# 直连
# robomaster.config.LOCAL_IP_STR = "192.168.2.1"
# ep_robot.initialize(conn_type='ap')
ep_chassis= ep_robot.chassis
ep_camera = ep_robot.camera
ep_camera.start_video_stream(display=True)

# 开启键盘监听事件
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

spd = 0
img_index = 1
datas = {"image_path":[],"angle":[],"speed":[],"direct":[]}
while not o_pressed:
    img = ep_camera.read_cv2_image()
    # 更新时间速度
    if w_pressed:
        tm = round(find_index(ac,spd)/10 + 0.1,6)
        spd = round(ac[int(10*tm)],6)
    else:
        tm = round(find_index(de,spd)/10 + 0.1,6)
        spd = round(de[int(10*tm)],6)
    # 按住p记录数据
    if n_pressed:
        datas['direct'].append([1,0,0])
        try:
            datas['angle'].append(round(calculate_rotation_angle(last_img, img), 2))
        except:
            datas['angle'].append(0)
        image_path = "train/images/"+str(img_index)+".jpg"
        cv2.imwrite(image_path, img)
        datas['image_path'].append(image_path)
        datas['speed'].append(spd)
        img_index += 1
    last_img = img
    print(spd)
    time.sleep(0.1)
d = pd.DataFrame(datas)
d.to_csv('train/dataset.csv', index=False)

ep_robot.close()
listener.stop()
