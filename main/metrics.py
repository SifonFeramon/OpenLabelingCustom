from collections import defaultdict
import utils as u
import os
import numpy as np


def side_coords(box):
    return np.array([
        box[0] - box[2] / 2,
        box[1] - box[3] / 2,
        box[0] + box[2] / 2,
        box[1] + box[3] / 2,
    ])


# input box: [x left, y top, x right, y bottom]
def box_area(box):
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)


# input arrays: [x left, y top, x right, y bottom]
def IOU(a, b):
    intersection = [
        max(a[0], b[0]),
        max(a[1], b[1]),
        min(a[2], b[2]),  
        min(a[3], b[3]),
    ] 
    intersection_area = box_area(intersection)
    full_area = box_area(a) + box_area(b) - intersection_area
    return box_area(intersection) / full_area


# если у предсказания наивысший ретинг среди нескольких ground меток,
# то выбираем наилучшее.
def find_best_fit_ground(intersections, pred_index):
    max_iou = -1
    max_iou_v_index = -1
    for v, v_index in zip(intersections, range(len(intersections))):
        best_predict = v[1][0]
        predicted_iou = best_predict[1]
        predicted_index = best_predict[2]   
        if predicted_index == pred_index and max_iou < predicted_iou:
            max_iou_v_index = v_index
            max_iou = predicted_iou

    return max_iou_v_index


# для каждой истинной метки находим IOU с каждой предсказанной меткой
# input matrices: [[class, x center, y center, width, height]]
# output: [ground class, [(class, iou, index)]]
def find_iou_intersections(ground_truths, predictions):
    intersections = []
    # для каждой истинной метки находим IOU с каждой предсказанной меткой
    for ground, g_index in zip(ground_truths, range(len(ground_truths))):
        intersections.append([ground[0], []])
        for predicted, p_index in zip(predictions, range(len(predictions))):
            intersections[g_index][1].append([predicted[0], IOU(ground[1:], predicted[1:]), p_index])
    return intersections


def remove_duplicate_predictions(intersections, best_ground_index, predicted_index):
    for v, v_index in zip(intersections, range(len(intersections))):
        if v_index != best_ground_index:
            filtered_list = []
            for item in v[1]:
                if item[2] != predicted_index:
                    filtered_list.append(item)
            v[1] = filtered_list


# input matrix: [[class, x center, y center, width, height]]
# out matrix: [[class, x left, y top, x right, y bottom]]
def convert_config_matrix(matrix):
    sided= np.apply_along_axis(side_coords, 1, matrix[:, 1:])
    return np.concatenate([matrix[:, 0:1], sided], axis=1)


# input matrices: [[class, x center, y center, width, height]]
# output matrix: [(class, precision, recall, IoU)]
def calc_metrics(predictions, ground_truths):
    predictions = convert_config_matrix(predictions)
    ground_truths = convert_config_matrix(ground_truths)
    intersections = find_iou_intersections(ground_truths, predictions)

    # сортируем предсказания по IOU
    for l in intersections:
        l[1].sort(key=lambda x: x[1], reverse=True)
    
    # статистика для каждого класса объектвов
    # {class, (nTP, nFP, nFN, sum IoU, iou count)}
    stats = defaultdict(lambda:[0, 0, 0, 0, 0])

    # выявление соответствий между предсказаниями и метками
    for predicted, pred_index in zip(predictions, range(len(predictions))):
        best_ground_index = find_best_fit_ground(intersections, pred_index)

        # если предскзание нигде не получило высший рейтинг, значит это дубликат
        # дубликты считаюстся ложноположительными предсказаниями
        if best_ground_index == -1:
            stats[predicted[0]][1] += 1

        remove_duplicate_predictions(intersections, best_ground_index, pred_index)
    
    for v in intersections:
        cls_stats = stats[v[0]]
        # если предсказаний нет, то ложноотрицательный
        if len(v[1]) == 0:
            cls_stats[2] += 1
            continue

        best_predict = v[1][0]

        # показания для расчета среднего IOU
        cls_stats[3] += best_predict[1]
        cls_stats[4] += 1

        # если пересечение маленькое, то ложноположительный
        if best_predict[1] < 0.5:
            cls_stats[1] += 1
        # если класс совпал, то истиноположительный
        elif best_predict[0] == v[0]:
            cls_stats[0] += 1
        # если класс предсказан неверно, то ложноотрицательный
        else:
            cls_stats[2] += 1

    # [(class, precision, recall, IoU)]
    final = np.empty(shape=(len(stats), 4), dtype=float)

    # вычислям финальные метрики
    i = 0
    for cls, stat in stats.items():
        final[i][0] = cls
        final[i][1] = stat[0] / (stat[0] + stat[1])
        final[i][2] = stat[0] / (stat[0] + stat[2])
        final[i][3] = stat[3] / stat[4]
        i += 1

    return final


# out matrix: [[class, x center, y center, width, height]
def read_config_matrix(path):
    out = []

    file = open(path)
    for line in file.readlines():
        out.append([])
        words = line.split()
        for word in words:
            out[-1].append(float(word))
    file.close()
    return np.array(out)


def calc_file_metrics(ground_dir, predict_dir, filename):
    ground = read_config_matrix(os.path.join(ground_dir, filename))
    predict = read_config_matrix(os.path.join(predict_dir, filename))
    return calc_metrics(predict, ground)

# out [(precision, recall, IoU)]
def filter_cls_metrics(metrics, cls):
    cls_metrics = []
    for file_metrics in metrics:
        file_metrics = file_metrics[1]
        for cls_metric in file_metrics:
            if cls_metric[0] == cls:
                cls_metrics.append(cls_metric[1:])
                break
    return np.array(cls_metrics)

# out [(precision, recall)] 
def interpolated_precision(metrics):
    if len(metrics) == 0:
        return []

    # сортируем метрики по полноте
    metrics = metrics[metrics[:, 2].argsort()]
    
    # [(percision, recall)]
    results = [[metrics[0][0], metrics[0][1]]]

    for metric in metrics[1:]:
        precision = metric[0]
        recall = metric[1]

        if results[-1][1] != recall:
            results.append((results[-1][0], results[-1][1]))
            results[-1][1] = recall
            results[-1][0] = -1
        else:
            results[-1][0] = max(results[-1][0],  precision)

    return results

def find_precision_for_recall(interpolated, recall):
    for i in range(1, len(interpolated)):
        if interpolated[i][1] > recall:
            return interpolated[i - 1][0]
    return interpolated[0][0]


def calculate_AP(metrics):
    interpolated = interpolated_precision(metrics)
    if(len(interpolated) == 0):
        return 1.0

    recall_segments = np.arange(0.0, 1.1, 0.1)

    sum = 0.0
    for recall in recall_segments:
        sum += find_precision_for_recall(interpolated, recall)
    return sum / len(recall_segments)

def calculate_mAP(ground_dir, predict_dir, cls_list):
    def load_f(file):
        return calc_file_metrics(ground_dir, predict_dir,  os.path.basename(file))

    metrics = u.load_from_directory(predict_dir, load_f, u.is_file_txt)
    sum = 0.0
    for cls in cls_list:
        cls_metrics = filter_cls_metrics(metrics, cls)
        sum += calculate_AP(cls_metrics)
    return sum / len(cls_list)


# main_dir = 'C:/Users/smirn/Documents/RSM/OpenLabelingCustom/main/'
# ground_dir = main_dir + 'output/YOLO_darknet'
# predict_dir = main_dir + 'predicts'
# print(calculate_mAP(ground_dir, predict_dir, [0, 1]))