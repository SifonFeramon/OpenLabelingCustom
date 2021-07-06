from os import listdir
from os.path import isfile, join
from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove
path = "/home/nikolay/OpenLabeling/main/output/YOLO_darknet"
onlyfiles = [f"{path}/{f}" for f in listdir(f"{path}") if f.endswith(".txt")]
class_n = input("Enter old class number: ")
class_nn = input("Enter new class number: ")
for file in onlyfiles:
    print(file)
    with open(f"{file}.new",'w') as new_file:
        with open(file) as old_file:
            for line in old_file:
                if line.startswith(f"{class_n}"):
                    new_file.write(f"{class_nn}{line[len(class_n):]}")
                else:
                    new_file.write(line)
    remove(file)
    move(f"{file}.new", file)

