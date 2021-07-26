import os
import cv2
import utils as u


def test_show_image(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)


# возвращает массив opencv изображений, полученных из видео
def convert_video_to_images(path):
    result = []
    video_cap = cv2.VideoCapture(path)
    n_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in range(n_frames):
        ret, frame = video_cap.read()
        if ret == False:
            break
        result.append(frame)
    video_cap.release()
    return result



def video_images_folder(video_path):
    file_path, file_extension = os.path.splitext(video_path)
    # append extension to avoid collision of videos with same name
    # e.g.: `video.mp4`, `video.avi` -> `video_mp4/`, `video_avi/`
    file_extension = file_extension.replace('.', '_')
    file_path += file_extension
    return file_path 


# возвращает массив opencv изображений, полученных из видео. Кэширует их в папку на диск.
def get_video_images(video_path, img_format=".jpg"):
    # create folder to store images (if video was not converted to images already)
    file_path = video_images_folder(video_path)
    video_name_ext = os.path.basename(file_path)

    def save_image_as_frame(f_i):
        frame_name = '{}_{}{}'.format(video_name_ext, f_i[1], img_format)
        frame_path = os.path.join(file_path, frame_name)
        return frame_path, cv2.imwrite(frame_path, f_i[0])

    if not os.path.exists(file_path) or u.is_folder_empty(file_path):
        os.makedirs(file_path, exist_ok=True)
        images = convert_video_to_images(video_path)
        image_indices = zip(images, range(len(images)))
        return video_name_ext, u.process_parallel(image_indices, save_image_as_frame)
    else:
        return video_name_ext, u.load_from_directory(file_path, cv2.imread, u.is_file_image, True)


# https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf