import torch
import numpy as np
import os

'''
Event data loader
Transform the original event data to .pt file for E2NeRF training
'''
def load_event_data_noisy(data_path):
    event_map = np.zeros((100, 4, 800, 800))    # 100 views of input, 4 event bin (b in the paper) for each view, resolution of 800*800

    for i in range(0, 200, 2):
        file = os.path.join(data_path, "r_{}/v2e-dvs-events.txt".format(i))
        fp = open(file, "r")

        print("Processing data_path_{}".format(i))
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

            if t > counter * 0.04 + 0.01:
                counter += 1
                if counter >= 5:
                    break

            if p == 0:
                event_map[int(i / 2)][counter - 1][y][x] -= 1
            else:
                event_map[int(i / 2)][counter - 1][y][x] += 1
    return event_map

input_data_path = "../blender-v2e-synthetic-events/lego/"
output_data_path = "../blender-v2e-synthetic-events/lego/events.pt"

if __name__ == '__main__':
    events = load_event_data_noisy(input_data_path)
    events = torch.tensor(events).view(100, 4, 640000)
    torch.save(events, output_data_path)