import os.path
import cv2
import torch
import numpy as np
from dv import AedatFile

'''
preload the aedat file
transform the video into separate blurry frames
transform the corresponding event into .pt file for E2NeRF training
'''

def trans(x):
    if x:
        return 1
    else:
        return 0

def load_events(file, time, views_num, inter_num):
    '''
    according to the start exposure timestamp and end exposure timestamp of the blurry image to select the corresponding event
    output the .pt preloaded event data for E2NeRF training
    '''
    f = AedatFile(file)
    frame_num = 0
    event_map = np.zeros((views_num, inter_num - 1, 260, 346))

    for event in f["events"]:
        if time[frame_num][1] < event.timestamp:
            frame_num = frame_num + 1
            if frame_num >= views_num:
                break

        if time[frame_num][0] <= event.timestamp <= time[frame_num][1]:
            if event.polarity:
                event_map[frame_num][int((event.timestamp - time[frame_num][0]) / 25001)][event.y][event.x] += 1
            else:
                event_map[frame_num][int((event.timestamp - time[frame_num][0]) / 25001)][event.y][event.x] -= 1
    return event_map


def load_frames(file, basedir, views_num, inter_num):
    '''
    preload the blurry frames in aedat4 file
    output the start exposure timestamp and end exposure timestamp of the corresponding blurry image
    '''
    global s
    f = AedatFile(file)
    sum = 0
    times = []
    for frame in f["frames"]:
        if sum % 10 == 0:
            cv2.imwrite(basedir + data_name + "/images/{0:03d}.jpg".format(s * inter_num), frame.image)
            times.append([frame.timestamp_start_of_exposure, frame.timestamp_end_of_exposure])
            s += 1
        sum += 1
        if s == views_num:
            break
    return times


basedir = "../davis-aedat4/"
data_name = "lego"
height = 260
width = 346
global s
s = 0

if __name__ == '__main__':
        events = []

        inter_num = 5       # The number of the event bin (b in paper) + 1
        views_num = 30      # The number of the views of the scene

        if not os.path.exists(basedir + data_name + "/images"):
            os.mkdir(basedir + data_name + "/images")

        file = basedir + data_name + ".aedat4"

        times = load_frames(file, basedir, views_num, inter_num)
        events.append(load_events(file, times, views_num, inter_num))
        events = np.concatenate(events)
        events = torch.tensor(events).view(-1, inter_num - 1, 89960)
        torch.save(events, basedir + data_name + "/events.pt")






