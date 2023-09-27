from util import *
import math
import torch.distributions as tdist


def tone_mapping(image: torch.Tensor, debug=False):
    image = torch.clamp(image, min=0.0, max=1.0)
    out = 3.0 * torch.pow(image, 2) - 2.0 * torch.pow(image, 3)
    if debug:
        single_real_image_display(out, "Image_After_Tone_Mapping")
    return out


def gamma_compression(image: torch.Tensor, debug=False):
    out = torch.pow(torch.clamp(image, min=1e-8), 1 / 2.2)
    if debug:
        single_real_image_display(out, "Image_After_Gamma_Compression")
    return out


def color_correction(image: torch.Tensor, ccm, debug=False):
    shape = image.size()
    image = torch.reshape(image, [-1, 3])
    image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
    out = torch.reshape(image, shape)
    if debug:
        single_real_image_display(out, "Image_After_Color_Correction")
    return out


def demosaic(image: torch.Tensor, debug=False):
    """ Reference Code: https://github.com/Chinmayi5577/Demosaicing """
    shape = image.size()
    image = image.numpy()
    out_np = np.zeros((shape[0] * 2, shape[1] * 2, 3), dtype=np.float32)

    out_np[0::2, 0::2, 0] = image[:, :, 0]
    out_np[0::2, 1::2, 1] = image[:, :, 1]
    out_np[1::2, 0::2, 1] = image[:, :, 2]
    out_np[1::2, 1::2, 2] = image[:, :, 3]

    for i in range(1, shape[0] * 2 - 1):
        for j in range(1, shape[1] * 2 - 1):
            if i % 2 == 1 and j % 2 == 0:  # case when green pixel is present with red on top and bottom
                out_np[i][j][0] = (out_np[i - 1][j][0] + out_np[i + 1][j][0]) / 2  # red
                out_np[i][j][2] = (out_np[i][j - 1][2] + out_np[i][j + 1][2]) / 2  # blue
            elif i % 2 == 0 and j % 2 == 0:  # case when red pixel is present
                out_np[i][j][1] = (out_np[i - 1][j][1] + out_np[i][j + 1][1] +
                                   out_np[i + 1][j][1] + out_np[i][j - 1][1]) / 4
                out_np[i][j][2] = (out_np[i - 1][j - 1][2] + out_np[i + 1][j - 1][2] +
                                   out_np[i - 1][j + 1][2] + out_np[i + 1][j + 1][2]) / 4
            elif i % 2 == 0 and j % 2 == 1:  # case when green pixel is present with blue on top and bottom
                out_np[i][j][0] = (out_np[i][j + 1][0] + out_np[i][j - 1][0]) / 2
                out_np[i][j][2] = (out_np[i + 1][j][2] + out_np[i - 1][j][2]) / 2
            else:  # case when blue pixel is present
                out_np[i][j][0] = (out_np[i - 1][j - 1][0] + out_np[i - 1][j + 1][0] +
                                   out_np[i + 1][j + 1][0] + out_np[i + 1][j - 1][0]) / 4
                out_np[i][j][1] = (out_np[i - 1][j][1] + out_np[i][j + 1][1] +
                                   out_np[i + 1][j][1] + out_np[i][j - 1][1]) / 4

    last_row = shape[0] * 2 - 1
    for j in range(1, shape[1] * 2 - 1):
        if j % 2 == 0:  # case when red pixel is present on first row or green pixel on last row
            out_np[0][j][1] = (out_np[0][j - 1][1] + out_np[0][j + 1][1] + out_np[1][j][1]) / 3
            out_np[0][j][2] = (out_np[1][j - 1][2] + out_np[1][j + 1][2]) / 2
            out_np[last_row][j][0] = out_np[last_row - 1][j][0]
            out_np[last_row][j][2] = (out_np[last_row][j - 1][2] + out_np[last_row][j + 1][2]) / 2
        else:  # case when green pixel is present on first row or blue pixel on last row
            out_np[0][j][0] = (out_np[0][j - 1][0] + out_np[0][j + 1][0]) / 2
            out_np[0][j][2] = out_np[1][j][2]
            out_np[last_row][j][0] = (out_np[last_row - 1][j - 1][0] + out_np[last_row - 1][j + 1][0]) / 2
            out_np[last_row][j][1] = (out_np[last_row][j - 1][1] + out_np[last_row][j + 1][1] +
                                      out_np[last_row - 1][j][1]) / 3

    last_column = shape[1] * 2 - 1
    for i in range(1, shape[0] * 2 - 1):
        if i % 2 == 0:  # case when red pixel is present on first column or green pixel on last column
            out_np[i][0][1] = (out_np[i - 1][0][1] + out_np[i + 1][0][1] + out_np[i][1][1]) / 3
            out_np[i][0][2] = (out_np[i - 1][1][2] + out_np[i + 1][1][2]) / 2
            out_np[i][last_column][0] = out_np[i][last_column - 1][0]
            out_np[i][last_column][2] = (out_np[i - 1][last_column][2] + out_np[i + 1][last_column][2]) / 2
        else:  # case when green pixel is present on first column or blue pixel on last column
            out_np[i][0][0] = (out_np[i - 1][1][0] + out_np[i + 1][1][0]) / 2
            out_np[i][0][2] = out_np[i][1][2]
            out_np[i][last_column][0] = (out_np[i - 1][last_column - 1][0] + out_np[i + 1][last_column - 1][0]) / 2
            out_np[i][last_column][1] = (out_np[i - 1][last_column][1] + out_np[i + 1][last_column][1] +
                                         out_np[i][last_column - 1][1]) / 3

    out_np[0][0][1] = (out_np[0][1][1] + out_np[1][0][1]) / 2
    out_np[0][0][2] = out_np[1][1][2]
    out_np[0][last_column][0] = out_np[0][last_column - 1][0]
    out_np[0][last_column][2] = out_np[1][last_column][2]
    out_np[last_row][0][0] = out_np[last_row - 1][0][0]
    out_np[last_row][0][2] = out_np[last_row][1][2]
    out_np[last_row][last_column][0] = out_np[last_row - 1][last_column - 1][0]
    out_np[last_row][last_column][1] = (out_np[last_row - 1][last_column][1] +
                                        out_np[last_row][last_column - 1][1]) / 2

    out = torch.from_numpy(out_np)
    out = torch.clamp(out, min=0.0, max=1.0)
    if debug:
        single_real_image_display(out, "Image_After_Demosaic")
    return out


def white_balance(image: torch.Tensor, red_gain=1.9, blue_gain=1.5, debug=False):
    f_red = image[:, :, 0] * red_gain
    f_blue = image[:, :, 3] * blue_gain
    out = torch.stack((f_red, image[:, :, 1], image[:, :, 2], f_blue), dim=-1)
    out = torch.clamp(out, min=0.0, max=1.0)
    if debug:
        single_raw_image_display(out, "Image_After_White_Balance")
    return out


def digital_gain(image: torch.Tensor, iso=800, debug=False):
    out = image * iso
    if debug:
        tmp = torch.clamp(out, min=0.0, max=1.0)
        single_raw_image_display(tmp, "Image_After_Digital_Gain")
    return out


def exposure_time_accumulate(image: torch.Tensor, exposure_time=10, debug=False):
    out = image * exposure_time
    if debug:
        tmp = torch.clamp(out, min=0.0, max=1.0)
        single_raw_image_display(tmp, "Image_After_Exposure_Time_Accumulation")
    return out


def add_read_noise(image: torch.Tensor, iso=800, debug=False):
    image *= (1023 - 64)
    r_channel = image[:, :, 0]
    gr_channel = image[:, :, 1]
    gb_channel = image[:, :, 2]
    b_channel = image[:, :, 3]

    R0 = {'R': 0.300575, 'G': 0.347856, 'B': 0.356116}
    R1 = {'R': 1.293143, 'G': 0.403101, 'B': 0.403101}

    r_noise_sigma_square = (iso / 100.) ** 2 * R0['R'] + R1['R']
    gb_noise_sigma_square = (iso / 100.) ** 2 * R0['G'] + R1['G']
    gr_noise_sigma_square = (iso / 100.) ** 2 * R0['G'] + R1['G']
    b_noise_sigma_square = (iso / 100.) ** 2 * R0['B'] + R1['B']

    r_samples = tdist.Normal(loc=torch.zeros_like(r_channel), scale=math.sqrt(r_noise_sigma_square)).sample()
    gr_samples = tdist.Normal(loc=torch.zeros_like(gr_channel), scale=math.sqrt(gb_noise_sigma_square)).sample()
    gb_samples = tdist.Normal(loc=torch.zeros_like(gb_channel), scale=math.sqrt(gr_noise_sigma_square)).sample()
    b_samples = tdist.Normal(loc=torch.zeros_like(b_channel), scale=math.sqrt(b_noise_sigma_square)).sample()

    r_channel += r_samples
    gr_channel += gr_samples
    gb_channel += gb_samples
    b_channel += b_samples

    out = torch.stack((r_channel, gr_channel, gb_channel, b_channel), dim=-1)
    out /= (1023 - 64)
    out = torch.clamp(out, min=0.0, max=1.0)

    if debug:
        single_raw_image_display(out, "Image_After_Add_Read_Noise")
    return out


def add_shot_noise(image: torch.Tensor, debug=False):
    image *= (1023 - 64)
    r_channel = image[:, :, 0]
    gr_channel = image[:, :, 1]
    gb_channel = image[:, :, 2]
    b_channel = image[:, :, 3]

    S = {'R': 0.343334/2, 'G': 0.348052/2, 'B': 0.346563/2}

    r_noise_sigma_square = (S['R'] / 100) * r_channel
    gb_noise_sigma_square = (S['G'] / 100) * gr_channel
    gr_noise_sigma_square = (S['G'] / 100) * gb_channel
    b_noise_sigma_square = (S['B'] / 100) * b_channel

    # print(r_noise_sigma_square)
    # print(torch.sqrt(r_noise_sigma_square).shape, r_channel.shape)

    r_samples = tdist.Normal(loc=torch.zeros_like(r_channel), scale=torch.sqrt(r_noise_sigma_square)).sample()
    gr_samples = tdist.Normal(loc=torch.zeros_like(gr_channel), scale=torch.sqrt(gb_noise_sigma_square)).sample()
    gb_samples = tdist.Normal(loc=torch.zeros_like(gb_channel), scale=torch.sqrt(gr_noise_sigma_square)).sample()
    b_samples = tdist.Normal(loc=torch.zeros_like(b_channel), scale=torch.sqrt(b_noise_sigma_square)).sample()

    r_channel += r_samples
    gr_channel += gr_samples
    gb_channel += gb_samples
    b_channel += b_samples

    out = torch.stack((r_channel, gr_channel, gb_channel, b_channel), dim=-1)
    out /= (1023 - 64)
    out = torch.clamp(out, min=0.0, max=1.0)

    if debug:
        single_raw_image_display(out, "Image_After_Add_Shot_Noise")
    return out


def process(image: torch.Tensor, args, debug=False):
    # Followings are ISP Steps
    # ISP-step -7: the exposure time accumulation
    image = exposure_time_accumulate(image, exposure_time=args.output_exposure, debug=debug)
    image = add_shot_noise(image, debug)
    # ISP-step -6: the digital gain with read noise
    image = digital_gain(image, iso=args.output_iso, debug=debug)
    image = add_read_noise(image, iso=args.output_iso, debug=debug)
    # ISP-step -5: the white balance
    image = white_balance(image, red_gain=args.red_gain, blue_gain=args.blue_gain, debug=debug)
    # ISP-step -4: the demosaicing
    image = demosaic(image, debug)
    # ISP-step -3: the color correction
    image = color_correction(image, args.cam2rgb, debug)
    # ISP-step -2: the gamma compression
    image = gamma_compression(image, debug)
    # ISP-step -1: the tone mapping
    image = tone_mapping(image, debug)
    return image
