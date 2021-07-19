from collections import defaultdict
import utils as u

def count_classes(folder):
    counting = defaultdict(int) 

    for f in u.list_folder(folder, u.is_file_txt):
        f = open(f, 'r')
        lines = f.readlines()
        for line in lines:
            cls = int(line.split(None, 1)[0])
            counting[cls] += 1
        f.close()

    return counting