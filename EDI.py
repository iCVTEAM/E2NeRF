import torch
import cv2
import math


# -------------------EDI MODEL----------------------
def EDI_rgb(data_path, frame_num, bin_num, events):
    event_sum = torch.zeros(260, 346)
    EDI = torch.ones(260, 346)

    for i in range(bin_num):
        event_sum = event_sum + events[frame_num][i]
        EDI = EDI + torch.exp(0.3 * event_sum)

    EDI = torch.stack([EDI, EDI, EDI], axis=-1)
    img = (bin_num + 1) * blurry_image / EDI
    img = torch.clamp(img, max=255)
    cv2.imwrite(data_path + "images_for_colmap/{0:03d}.jpg".format(frame_num * (bin_num + 1)), img.numpy()) # save the first deblurred image

    offset = torch.zeros(260, 346)
    for i in range(bin_num):
        offset = offset + events[frame_num][i]
        imgs = img * torch.exp(0.3 * torch.stack([offset, offset, offset], axis=-1))
        cv2.imwrite(data_path + "images_for_colmap/{0:03d}.jpg".format(frame_num * (bin_num + 1) + 1 + i), imgs.numpy()) # save the rest of the deblurred images

threshold = 0.3
bin_num = 4
view_num = 30
data_path = "./data/"
events = torch.loadtxt(data_path + "events.pt").view(view_num, bin_num, 260, 346) # load the preprocessed events in .pt file

for i in range(view_num):
    blurry_image = torch.tensor(cv2.imread(data_path + "images/{0:03d}.jpg".format(i * (bin_num + 1))), dtype=torch.float) # load the target blurry image
    EDI_rgb(data_path, i, bin_num, events)



