import cv2

class ObjectDetector(object):
    def __init__(self, graph_path, score_threshold, objIds):
        pass

    def __init__(self):
        self.net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=(416, 416), scale=1/255)
        self.CONFIDENCE_THRESHOLD = 0
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def detect(self, frame):
        class_indices, scores, boxes = self.model.detect(frame, 0.1, 0.1)

        return boxes, scores, class_indices

    def detect2(self, frame):
        frame = frame[100:-100, 100:-100]


if __name__ == "__main__":
    cap = cv2.VideoCapture("/home/nikolay/git_builds/OpenLabeling/main/parkovka.mp4")
    detector = ObjectDetector()
    while (cv2.waitKey(0) not in [27, ord('q')]):
        _, res = cap.read()
        boxes, scores, class_indices = detector.detect(res)
        for i in range(len(scores)):
            box = boxes[i]
            if class_indices[i] == 0:
                prob = scores[i]
                cv2.rectangle(res, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0,255,255), 2)
                cv2.putText(res,f"{prob}", (box[0] + box[2]//2, box[1] + box[3]//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow("frame", res)

