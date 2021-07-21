import os
import re
import multiprocessing as mp
from joblib import Parallel, delayed


def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in _nsre.split(s)]


def get_sorted_dir_files(dir):
    files = sorted(os.listdir(dir), key=natural_sort_key)
    return [os.path.join(dir, f) for f in files if not os.path.isdir(f)]


def list_folder(path, f=os.path.isfile, need_sort=False):
    result = []
    files = os.listdir(path)
    if need_sort:
        files = sorted(files, key=natural_sort_key)

    for name in files:
        file = os.path.join(path, name)
        if(f(file)):
            result.append(file)
    return result


# возвращает массив(файл, результат), при которых функция вернула значение, дающее True
def load_from_directory(dir, load_f, filter_f=os.path.isfile, sorted=False):
    files = list_folder(dir, filter_f, sorted)
    results = process_parallel(files, load_f)

    filtered_result = []
    for f, r in zip(files, results):
        if len(r): filtered_result.append((f, r))

    return filtered_result 



def is_folder_empty(folder):
    return not any(os.scandir(folder))


def is_file_image(path):
    return os.path.isfile(path) and path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))


def is_file_video(path):
    return os.path.isfile(path) and path.lower().endswith(('.mp4', '.avi'))


def is_file_txt(path):
    return os.path.isfile(path) and path.lower().endswith(('.txt'))


def process_parallel(data, f):
    num_cores = mp.cpu_count()
    results = Parallel(n_jobs=num_cores - 2)(delayed(f)(d) for d in data)
    return [res for res in results if res is not None]

