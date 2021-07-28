import cv2
import sys

parts = 2
vname = sys.argv[1]
cap = cv2.VideoCapture(vname)
name, ext = vname.split('.')
print(name, ext)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
writers = []
for i in range(1, parts +1):
    writers.append(cv2.VideoWriter(f"{name}{i}.mp4", fourcc, 30, (1280, 960)))
maxw = len(writers)
i = 0
while (cv2.waitKey(1) != 27):
    success, frame = cap.read()
    if not success:
        break
    if True: #i > parts*50:
        if i%parts%2 == 1:
            frame = cv2.flip(frame, 1)
        writers[i%parts].write(frame)
    cv2.imshow("frame", frame)
    i = i + 1

for w in writers:
    w.release()

