from colorizers import *
import image_load
import warnings
import os
import numpy as np
import cv2
import torch.cuda as cuda

has_cuda = bool(cuda.device_count())
colorizer_siggraph17 = siggraph17(pretrained=True).eval()
if has_cuda:
    colorizer_siggraph17.cuda()

def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img

def colorize_image(input, output):
    warnings.filterwarnings("ignore")
    img = load_img(input)
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
    if has_cuda:
        tens_l_rs = tens_l_rs.cuda()
    out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())
    warnings.filterwarnings("default")
    result = convert(out_img_siggraph17, 0, 255, np.uint8)[:, :, ::-1]
    cv2.imwrite(output, result)
    return result

def get_colorized_images(files):
    folder = os.path.dirname(files[0])
    out_folder = os.path.join(folder, os.path.basename(folder) + "_color")
    os.makedirs(out_folder, exist_ok=True)

    def save_colorized(file):
        colorized_name = os.path.join(out_folder, os.path.basename(file))
        image = cv2.imread(colorized_name)
        if image is not None:
            return image
        else:
            print("colorizing image:", file)
            return colorize_image(file, colorized_name)

    return image_load.process_parallel(files, save_colorized)