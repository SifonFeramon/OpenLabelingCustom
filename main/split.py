import os
import shutil


def copy_each(src_dir, dst_dir, each):
    i = 0
    for file in os.listdir(src_dir):
        if file.endswith('.jpg'):
            if i % each == 0:
                file = os.path.join(src_dir, file)
                shutil.copy2(file, dst_dir)
                desc_file = file.split('.')[0]
                shutil.copy2(desc_file + ".txt", dst_dir)
            i += 1


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
