import argparse
from process import *
from unprocess import *
import cv2 as cv
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='../blender-synthetic-images/chair/r_0/', help='input frames dir')
parser.add_argument('--output_name', type=str, default='../blender-synthetic-images/chair/r_0.png', help='output frame name')
parser.add_argument('--scale_factor', type=int, default=18, help='convert scale_factor frames to 1 frame with blur, note that scale_factor must be divide by total number of frames of input')
parser.add_argument('--input_exposure', type=float, default=10, help='base exposure time of input frames in microsecond')
parser.add_argument('--input_iso', type=int, default=50, help='Assumed ISO for input data')
parser.add_argument('--output_iso', type=int, default=50, help='Expected ISO for output')
parser.add_argument('--debug', dest='debug', action='store_true', default=False, help='Debug mode')
parser.add_argument('--show_effect', dest='show_effect', action='store_true', default=False,
                    help='Show the initial image and final image')
args = parser.parse_args()

if __name__ == '__main__':
    # if file_cnt accum to scale_factor, output it and file_cnt reset to 0
    file_cnt = 0
    file_tot_cnt = 0
    # acc_img: img accum to acc_img
    acc_img = 0.
    # get file list
    # input_list = sorted(os.listdir(args.input_dir))
    input_list = os.listdir(args.input_dir)
    input_list.sort(key=lambda x: int(x[:-4]))
    # get image list
    img_list = []
    for input_file in input_list:
        if input_file.split('.')[1] == 'jpg' or input_file.split('.')[1] == 'png':
            img_list.append(input_file)

    # scan the input dir
    with tqdm(total=len(img_list), desc='process of file datasimul') as pbar:
        for filename in img_list:
            assert(filename.split('.')[1] == 'jpg' or filename.split('.')[1] == 'png')
            # Read the image
            img = imread_rgb(args.input_dir + filename)

            # make the image H W to even
            flagH, flagW = img.shape[0] % 2, img.shape[1] % 2
            img = cv.copyMakeBorder(img, flagH, 0, flagW, 0, cv.BORDER_WRAP)

            H, W, C = img.shape

            # Transform image from numpy.ndarray to torch.Tensor type
            img = torch.from_numpy(img)

            # Display the initial image
            if args.show_effect:
                single_8bit_image_display(img, "Initial")

            # Transform image from [0, 255] to [0, 1]
            img = image_8bit_to_real(img)

            # Fundamental Arguments Settings
            args.rgb2cam, args.cam2rgb = random_ccm()
            args.red_gain, args.blue_gain = random_gains()
            args.output_exposure = args.input_iso * args.input_exposure / args.output_iso

            # Invert ISP
            img = unprocess(image=img, args=args, debug=args.debug)

            # file_cnt accum
            file_cnt += 1
            file_tot_cnt += 1
            acc_img += img

            if(file_cnt == args.scale_factor):

                file_cnt = 0
                acc_img.clamp_(0., 1.)
                # ISP
                acc_img = process(image=acc_img, args=args, debug=args.debug)
                # Transform image from [0, 1] to [0, 255]
                acc_img = image_real_to_8bit(acc_img)

                # Display the final image
                if args.show_effect:
                    single_8bit_image_display(acc_img, "Final")

                # Transform image from torch.Tensor type to numpy.ndarray
                acc_img = acc_img.numpy()

                # Write the image
                rgba_img = cv2.cvtColor(acc_img, cv2.COLOR_BGR2RGBA)
                cv2.imwrite(args.output_name, rgba_img)
                acc_img = 0.

                pbar.update(args.scale_factor)

