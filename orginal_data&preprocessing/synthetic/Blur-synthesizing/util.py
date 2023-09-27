import cv2
import numpy as np
import torch
import numpy


def imread_rgb(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def imwrite_rgb(image, filename):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(filename, image)


def image_8bit_to_real(image_8bit: torch.Tensor):
    """ Covert image from [0, 255] to [0, 1] """
    return torch.div(image_8bit, 255)


def image_real_to_8bit(image_real: torch.Tensor):
    """ Covert image from [0, 1] to [0, 255] """
    image_real = torch.clamp(image_real, min=0.0, max=1.0)
    return torch.mul(image_real, 255).type(torch.uint8)


def single_8bit_image_display(image: torch.Tensor, description=None):
    """ Display a single [0, 255] image """
    image = image.numpy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # cv2.imshow(description, image)
    # cv2.waitKey(0)
    cv2.imwrite('./resfig/ans.jpg', image)


    # cv2.destroyAllWindows()


def single_real_image_display(image: torch.Tensor, description=None):
    """ Display a single [0, 1] image """
    # image_real_to_8bit(image)
    image = image.numpy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow(description, image)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()


def single_8bit_image_display_numpy(image: numpy.ndarray, description=None):
    cv2.imshow(description, image)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()


def single_raw_image_display(image: torch.Tensor, description=None):
    shape = image.size()
    image = image.numpy()
    raw_in_rgb = numpy.zeros([shape[0] * 2, shape[1] * 2, 3], dtype=np.float32)
    raw_in_rgb[0::2, 0::2, 0] = image[:, :, 0]
    raw_in_rgb[0::2, 1::2, 1] = image[:, :, 1]
    raw_in_rgb[1::2, 0::2, 1] = image[:, :, 2]
    raw_in_rgb[1::2, 1::2, 2] = image[:, :, 3]
    raw_in_rgb = cv2.cvtColor(raw_in_rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow(description, raw_in_rgb)
    cv2.imwrite(description + ".jpg", raw_in_rgb * 255)
    cv2.waitKey(0)


def random_ccm():
    """Generates random RGB -> Camera color correction matrices."""
    # Takes a random convex combination of XYZ -> Camera CCMs.
    xyz2cams = [[[1.0234, -0.2969, -0.2266],
                 [-0.5625, 1.6328, -0.0469],
                 [-0.0703, 0.2188, 0.6406]],
                [[0.4913, -0.0541, -0.0202],
                 [-0.613, 1.3513, 0.2906],
                 [-0.1564, 0.2151, 0.7183]],
                [[0.838, -0.263, -0.0639],
                 [-0.2887, 1.0725, 0.2496],
                 [-0.0627, 0.1427, 0.5438]],
                [[0.6596, -0.2079, -0.0562],
                 [-0.4782, 1.3016, 0.1933],
                 [-0.097, 0.1581, 0.5181]]]
    num_ccms = len(xyz2cams)
    xyz2cams = torch.FloatTensor(xyz2cams)
    weights = torch.FloatTensor(num_ccms, 1, 1).uniform_(1e-8, 1e8)
    weights_sum = torch.sum(weights, dim=0)
    xyz2cam = torch.sum(xyz2cams * weights, dim=0) / weights_sum

    # Multiplies with RGB -> XYZ to get RGB -> Camera CCM.
    rgb2xyz = torch.FloatTensor([[0.4124564, 0.3575761, 0.1804375],
                                 [0.2126729, 0.7151522, 0.0721750],
                                 [0.0193339, 0.1191920, 0.9503041]])
    rgb2cam = torch.mm(xyz2cam, rgb2xyz)

    # Normalizes each row.
    rgb2cam = rgb2cam / torch.sum(rgb2cam, dim=-1, keepdim=True)

    cam2rgb = torch.inverse(rgb2cam)

    rgb2cam = torch.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    cam2rgb = torch.inverse(rgb2cam)


    return rgb2cam, cam2rgb


def random_gains():
    # Red and blue gains represent white balance.
    # red_gain = torch.FloatTensor(1).uniform_(1.9, 2.4)
    red_gain = torch.FloatTensor([2.15])
    # blue_gain = torch.FloatTensor(1).uniform_(1.5, 1.9)
    blue_gain = torch.FloatTensor([1.7])
    return red_gain, blue_gain
