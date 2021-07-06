import cv2

class ObjectDetector(object):
    def __init__(self, graph_path, score_threshold, objIds):
        pass

    def __init__(self):
        self.net = cv2.dnn.readNet("yolo-obj_trees_poles_best.weights", "yolo-obj_trees_poles.cfg")
        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=(512, 512), scale=1/255)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def detect(self, frame):
        class_indices, scores, boxes = self.model.detect(frame, 0.6, 0.9)
        return boxes, scores, class_indices


if __name__ == "__main__":
    cap = cv2.VideoCapture("/home/nikolay/darknet/test.avi")
    detector = ObjectDetector()
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    wr = cv2.VideoWriter(f"res.mp4", fourcc, 30, (204, 360))
    while (cv2.waitKey(1) not in [27, ord('q')]):
        _, res = cap.read()
        boxes, scores, class_indices = detector.detect(res)
        for i in range(len(scores)):
            #print(boxes[i])
            box = boxes[i]
            xmin = box[0]
            ymin = box[1]
            xmax = xmin + box[2]
            ymax= ymin + box[3]
            prob = scores[i]
            cnum = class_indices[i]
            if cnum in [0, 7, 14]:
                color = (0, 255, 255)
                text = "Невеста"
                if cnum == 0:
                    color = (255, 0, 0)
                    text = "Женихъ"
                cv2.rectangle(res, (xmin, ymin), (xmax, ymax), color, 1)
                cv2.putText(res, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_COMPLEX, 0.3, color, 1)
        wr.write(res)

        cv2.imshow("frame", res)


