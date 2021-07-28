import cv2
import os
from PIL import Image
import subprocess

directory = "data"
files = [directory + "/" + f for f in os.listdir(directory) if f.endswith(('png', 'jpeg', 'jpg', 'tiff'))]

def read_exif(filename : str):
    result = subprocess.run(['exiftool', filename], stdout=subprocess.PIPE).stdout.decode("utf-8")
    print(result)
    for line in result.splitlines():
        if 'Image Description' in line:
            description = line.replace('Image Description','').replace('.','').replace(':', '').strip()
            if description:
                results = description.split(';')
                for result in results:
                    if result:
                        ll = result.split(' ')
                        yield ll[0], int(ll[1]), int(ll[2]), int(ll[3]), int(ll[4])


indx = 0
while True:
    frame = cv2.imread(files[indx])
    for det in read_exif(files[indx]):
        print(det)
        img, x, y, w, h = det
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
    cv2.imshow("frame", frame)
    inp = cv2.waitKey(0)
    if inp == ord('q') or inp == 27:
        break
    if inp == ord('a') or inp == 81:
        indx-=1
    if inp == ord('d') or inp == 83:
        indx+=1
    if indx <= 0 or indx > len(files) - 1:
        indx = 0


