from util import *


def inv_tone_mapping(image: torch.Tensor, debug=False):
    image = torch.clamp(image, min=0.0, max=1.0)
    out = 0.5 - torch.sin(torch.asin(1.0 - 2.0 * image) / 3.0)
    if debug:
        single_real_image_display(out, "Image_After_Inv_Tone_Mapping")
    return out


def inv_gamma_compression(image: torch.Tensor, debug=False):
    out = torch.pow(torch.clamp(image, min=1e-8), 2.2)
    if debug:
        single_real_image_display(out, "Image_After_Inv_Gamma_Compression")
    return out


def inv_color_correction(image: torch.Tensor, ccm, debug=False):
    shape = image.size()
    image = torch.reshape(image, [-1, 3])
    image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
    out = torch.reshape(image, shape)
    if debug:
        single_real_image_display(out, "Image_After_Inv_Color_Correction")
    return out


def mosaic(image: torch.Tensor, debug=False):
    shape = image.size()
    red = image[0::2, 0::2, 0]
    green_red = image[0::2, 1::2, 1]
    green_blue = image[1::2, 0::2, 1]
    blue = image[1::2, 1::2, 2]
    # import pdb
    # pdb.set_trace()
    # maybe shape[0] or shape[1] is odd, that cannot be divide by 2!!!!!!!

    out = torch.stack((red, green_red, green_blue, blue), dim=-1)
    out = torch.reshape(out, (shape[0] // 2, shape[1] // 2, 4))
    if debug:
        single_raw_image_display(out, "Image_After_Mosaic")
    return out


def inv_white_balance(image: torch.Tensor, red_gain=1.9, blue_gain=1.5, threshold=0.9, debug=False):
    red = image[:, :, 0]
    alpha_red = (torch.max(red - threshold, torch.zeros_like(red)) / (1 - threshold)) ** 2
    f_red = torch.max(red / red_gain, (1 - alpha_red) * (red / red_gain) + alpha_red * red)
    blue = image[:, :, 3]
    alpha_blue = (torch.max(blue - threshold, torch.zeros_like(blue)) / (1 - threshold)) ** 2
    f_blue = torch.max(blue / blue_gain, (1 - alpha_blue) * (blue / blue_gain) + alpha_blue * blue)
    out = torch.stack((f_red, image[:, :, 1], image[:, :, 2], f_blue), dim=-1)
    if debug:
        single_raw_image_display(out, "Image_After_Inv_White_Balance")
    return out


def inv_digital_gain(image: torch.Tensor, iso=800, scale_factor=5, debug=False):
    # out = image / iso
    out = image / (iso * scale_factor)
    # out /= scale_factor
    if debug:
        single_raw_image_display(out, "Image_After_Inv_Digital_Gain")
    return out


def inv_exposure_time(image: torch.Tensor, exposure_time=10, debug=False):
    out = image / exposure_time
    if debug:
        single_raw_image_display(out, "Image_After_Inv_Exposure_Time")
    return out


def unprocess(image: torch.Tensor, args, debug=False):
    # Followings are Inversion-ISP Steps
    # Inv-step 1: the inversion of tone mapping
    image = inv_tone_mapping(image, debug)
    # Inv-step 2: the inversion of gamma compression
    image = inv_gamma_compression(image, debug)
    # Inv-step 3: the inversion of color correction
    image = inv_color_correction(image, args.rgb2cam, debug)
    # Inv-step 4: the mosaicing
    image = mosaic(image, debug)
    # Inv-step 5: the inversion of white balance
    image = inv_white_balance(image, red_gain=args.red_gain, blue_gain=args.blue_gain, debug=debug)
    #single_raw_image_display(image, "balance")
    # Inv-step 6: the inversion of digital gain
    image = inv_digital_gain(image, iso=args.input_iso, scale_factor=args.scale_factor, debug=debug)

    # Inv-step 7: split exposure time to 1ms
    image = inv_exposure_time(image, exposure_time=args.input_exposure, debug=debug)

    return image
