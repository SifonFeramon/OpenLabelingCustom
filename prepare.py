import pprint
from collections import Counter
from os.path import expanduser, getsize, exists
from pathlib import Path
from os import listdir, walk
from os.path import isfile, join
from shutil import copyfile, rmtree
from random import shuffle
from filecmp import cmp
import time
import sys
import cv2
from tqdm.auto import tqdm

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def prepare_yolo():
    home = expanduser("~")
    str_p = f'{home}/darknet/data/obj'
    path = Path(str_p)
    path.mkdir(parents=True, exist_ok=True)
    toolbar_width = 80

    print("reading classes list from a file")
    objnames = []
    with open(f"./main/class_list.txt", 'r') as f:
        objnames = f.readlines()
        objnames = [x.strip() for x in objnames]

    print("available obj names")
    for o in objnames:
        print(o)


    print("""copy txt files from labeling folder to darknet data folder""")
    txt_path = f'./main/output/YOLO_darknet'
    txt_files = [str(f) for f in listdir(txt_path) if isfile(join(txt_path, f)) and f.endswith('txt')]
    files_pref = {}
    for f in tqdm(txt_files):
        prefix = ""
        source = f"{txt_path}/{f}"
        with open(source) as ff:
            try:
                prefix = ff.readline()[0]
            except Exception as e:
                pass

        destination = f"{home}/darknet/data/obj/{str(prefix)}{f}"
        raw = f.replace(".txt", "")
        files_pref[raw] = f"{str(prefix)}{raw}.jpg"
        if exists(destination) and (getsize(source) == getsize(destination)):
            continue
        copyfile(source, destination)

    print("""copy image files from labeling folder to darknet data recursive""")
    jpg_filenames = []
    jpg_path = f'./main/input'
    start = time.time()
    copy_tasks = {}
    for root, subdirs, files in walk(jpg_path):
        for f in files:
            if f.endswith('jpg'):
                source = f"{root}/{f}"
                prefix = ""
                dest = files_pref.get(f.replace(".jpg", ""))
                destination = f"{home}/darknet/data/obj/{dest if dest is not None else f}"
                jpg_filenames.append(destination)
                if exists(destination) and (getsize(source) == getsize(destination)):
                    continue
                copy_tasks[source] = destination

    for source, destination in tqdm(copy_tasks.items()):
        copyfile(source, destination)
        if cv2.imread(destination) is None:
            jpg_filenames.remove(destination)
            print(f"broken file removed:{destination}")


    print("""creating file with obj names""")
    with open(f"{home}/darknet/data/obj.names", 'w') as f:
        for obj in objnames:
            f.write(f"{obj}\n")

    print("""creating file with obj data""")
    with open(f"{home}/darknet/data/obj.data", 'w') as f:
        f.write(
    f"""classes={len(objnames)}
    train  = data/train.txt
    valid  = data/valid.txt
    names = data/obj.names
    backup = backup/""")

    print("""creating all.txt file""")
    with open(f"{home}/darknet/data/all.txt", 'w') as f:
        for name in jpg_filenames:
            f.write(f"{name}\n".replace("nikolay", "user"))

    print("split files from all.txt to train.txt and valid.txt")
    part1 = []
    part2 = []
    with open(f"{home}/darknet/data/all.txt", 'r') as f:
        text = f.readlines()
        shuffle(text)
        ldelim = int(len(text)*0.1)
        part1 = text[:ldelim]
        part2 = text[ldelim:]


    cutting =  input("Do yo want equal size of each obj list?(Y/n)").lower() == 'y'


    with open(f"{home}/darknet/data/valid.txt", 'w') as f:
        for line in part1:
            f.write(f"{line}")

    cnt = Counter()
    for line in part2:
        cnt[line.rsplit('/', 1)[1][0]] += 1
    least_common = cnt.most_common(len(objnames))[-1]
    print(f"Меньше всего в датасете: {least_common}")

    print("splitting train set to equal chunks")
    """
    l_of_l = []
    for i in range(len(objnames)):
        l_of_l.append([x for x in part2 if x.rsplit('/', 1)[1].startswith(str(i))])
    for index, l in enumerate(l_of_l):
        l_of_l[index] = l[:int(least_common[1])]

    flat_list = [item for sublist in l_of_l for item in sublist]
    for item in flat_list:
        print(item)
    exit()
    """
    poles = [x for x in part2 if x.rsplit('/', 1)[1].startswith(str(0))]
    shuffle(poles)
    trees = [x for x in part2 if x.rsplit('/', 1)[1].startswith(str(1))]
    shuffle(trees)
    print("trees",len(trees))
    print("poles",len(poles))
    pole_ch = list(chunks(poles, len(poles) // 5))

    for txtn in range(1, 6):
        with open(f"{home}/darknet/data/train{txtn}.txt", 'w') as f:
            for line in pole_ch[txtn - 1]:
                f.write(f"{line}")
            shuffle(trees)
            for line in trees[:len(pole_ch[txtn -1])]:
                f.write(f"{line}")

    with open(f"{home}/darknet/data/train.txt", 'w') as f:
        for line in part2:
            f.write(f"{line}")



    print("""selecting cfg file""")
    nn_name = input("enter part of cfg name: ")
    cfg_files = [f for f in listdir(f"{home}/darknet/cfg/") if nn_name in f]
    if len(cfg_files) == 0:
        raise ValueError("bad name input, no such networks")
    elif len(cfg_files) == 1:
        nn_name = cfg_files[0]
    else:
        print("Please, select choice number:")
        for index, name in enumerate(cfg_files):
            print(f"[{index}] {name}")
        index = input("Enter choice number: ")
        nn_name = f"{cfg_files[int(index)]}"

    copyfile(f"{home}/darknet/cfg/{nn_name}", f"{home}/darknet/cfg/yolo-obj.cfg")

    print(f"editing yolo settings to yolo-obj.cfg")
    text = []
    with open(f"{home}/darknet/cfg/yolo-obj.cfg", 'r') as f:
        print(f"{home}/darknet/cfg/yolo-obj.cfg")
        text = f.readlines()

    max_batches = len(objnames)*2000
    nn_size = input("Enter network size: 384/416/608/640..")
    edit_filters = False
    text.reverse()
    for i, s in enumerate(text):
        if "[yolo]" in s:
            edit_filters = True
        if "subdivisions" in s:
            text[i] = "subdivisions=32\n"
        if "learning_rate" in s:
            text[i] = "learning_rate=0.0001\n"
        if 'max_batches' in s:
            text[i] = f"max_batches = {max_batches}\n"
        if s.startswith('steps'):
            text[i] = f"steps={0.8*max_batches},{0.9*max_batches}\n"
        if s.startswith('width'):
            text[i] = f"width={nn_size}\n"
        if s.startswith('height'):
            text[i] = f"height={nn_size}\n"
        if s.startswith('classes'):
            text[i] = f"classes={len(objnames)}\n"
        if edit_filters and s.startswith('filters'):
            text[i] = f"filters={(len(objnames) + 5)*3}\n"
            edit_filters = False
        if False:
            ...

    text.reverse()


    with open(f"{home}/darknet/data/yolo-obj.cfg", 'w') as f:
        for line in text:
            f.write(line)

    print(f"Saved to: {home}/darknet/data/yolo-obj.cfg")


    print("please, download: ")
    print("https://drive.google.com/file/d/1JKF-bdIklxOOVy-2Cr5qdvjgGpmGfcbp/view")
    print("and put it to darknet/data folder")
    print("To train yolo execute command:")
    print("./darknet detector train data/obj.data cfg/yolo-obj.cfg yolov4.conv.137 -dont_show -gpus 0,1 -map")

if __name__ == "__main__":
    prepare_yolo()

