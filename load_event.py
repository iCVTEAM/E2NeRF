import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2


# read event data to list
def load_event_data_v1(basedir):
    data_path = os.path.join(basedir, "event")
    event_list = []

    for i in range(0, 10, 2):

        file = os.path.join(data_path, "r_" + str(i) + "/v2e-dvs-events.txt")
        print(file)
        fp = open(file, "r")
        events = []
        event_block = []
        counter = 1

        for j in range(6):
            fp.readline()

        while True:
            line = fp.readline()
            if not line:
                events.append(event_block)
                break

            info = line.split()
            t = float(info[0])
            x = int(info[1])
            y = int(info[2])
            p = int(info[3])
            if t > counter * 0.001:
                events.append(event_block)
                event_block = []
                counter += 1
            event = [x, y, t, p]
            event_block.append(event)
        while counter < 20:
            counter += 1
            events.append([])
        event_list.append(events)
    return


# read event data to numpy
def load_event_data_v2(basedir):
    data_path = os.path.join(basedir, "event")
    event_map = np.zeros((100, 20, 800, 800), dtype=np.int)

    for i in range(0, 200, 2):

        file = os.path.join(data_path, "r_" + str(i) + "/v2e-dvs-events.txt")
        fp = open(file, "r")
        counter = 1

        for j in range(6):
            fp.readline()

        while True:
            line = fp.readline()
            if not line:
                break

            info = line.split()
            t = float(info[0])
            x = int(info[1])
            y = int(info[2])
            p = int(info[3])
            if t > counter * 0.001:
                counter += 1
            if p == 0:
                event_map[int(i / 2)][counter - 1][y][x] -= 1
            else:
                event_map[int(i / 2)][counter - 1][y][x] += 1

    return event_map
