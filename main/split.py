import os
import shutil
import glob


def copy_each(src_dir, dst_dir, each):
    i = 0
    for file in glob.iglob(src_dir + '**/*.jpg', recursive=True):
        if file.endswith('.jpg'):
            if i % each == 0:
                file = os.path.join(src_dir, file)
                shutil.copy2(file, dst_dir)
                desc_file = file.split('.')[0]
                shutil.copy2(desc_file + ".txt", dst_dir)
            i += 1


def copy_from_different(src_dir1, src_dir2, ext1, ext2, dst_dir1, dst_dir2):
    for file in os.listdir(src_dir1):
        if file.endswith(ext1):
            file1 = os.path.join(src_dir1, file)
            file2 = os.path.join(src_dir2, file.split('.')[0] + ext2)
            shutil.copy2(file1, dst_dir1)
            if os.path.exists(file2):
                shutil.copy2(file2, dst_dir2)
            else:
                open(file2, 'a').close()

def check_valid(src_dir, f1='.jpg', f2='.txt'):
    for file in os.listdir(src_dir):
        if file.endswith(f1):
            file = os.path.join(src_dir, file)
            desc_file = file.split('.')[0]
            if not os.path.exists(desc_file + f2):
                print(desc_file)


def copy_ext(src_dir, dst_dir, ext):
    for file in os.listdir(src_dir):
        if file.endswith(ext):
            file = os.path.join(src_dir, file)
            shutil.copy2(file, dst_dir)

copy_from_different(
   'c:/users/smirn/documents/rsm/openlabelingcustom/main/input/short_1/',
   'C:/Users/smirn/Documents/RSM/OpenLabelingCustom/main/output/YOLO_Darknet/',
   '.jpg', '.txt',
   'C:/Users/smirn/Documents/RSM/OpenLabelingCustom/main/images/',
   'C:/Users/smirn/Documents/RSM/OpenLabelingCustom/main/labels/'
   )
# copy_from_different(
#     'C:/Users/smirn/Documents/RSM/OpenLabelingCustom/main/output/YOLO_Darknet',
#     'C:/Users/smirn/Documents/RSM/OpenLabelingCustom/main/input/video_mp4',
#     '.txt', '.jpg', 'C:/Users/smirn/Documents/RSM/OpenLabelingCustom/main/short')
