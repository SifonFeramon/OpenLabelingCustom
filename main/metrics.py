from collections import defaultdict
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


# input matrices: [[class, x center, y center, width, height]]
# output: {class, (nTP, nFP, nFN)}
def confusion_counts(predictions, ground_truths):
    #[ground class, [(class, iou, index)]]
    intersections = []

    sided_predicts = np.apply_along_axis(side_coords, 1, predictions[:, 1:])
    predictions = np.concatenate([predictions[:, 0:1], sided_predicts], axis=1)
    sided_ground = np.apply_along_axis(side_coords, 1, ground_truths[:, 1:])
    ground_truths = np.concatenate([ground_truths[:, 0:1], sided_ground], axis=1)

    # для каждой истинной метки находим пересечения с каждой предсказанной меткой
    for ground, g_index in zip(ground_truths, range(len(ground_truths))):
        intersections.append([ground[0], []])
        for predicted, p_index in zip(predictions, range(len(predictions))):
            intersections[g_index][1].append([predicted[0], IOU(ground[1:], predicted[1:]), p_index])

    for l in intersections:
        l[1].sort(key=lambda x: x[1], reverse=True)
    
    for predicted, pred_index in zip(predictions, range(len(predictions))):
        # если у предсказания наивысший ретинг среди нескольких ground меток,
        # то выбираем наилучшее.
        max_iou = -1
        max_iou_v_index = -1
        for v, v_index in zip(intersections, range(len(intersections))):
            v = v[1]
            predicted_iou = v[0][1]
            predicted_index = v[0][2]   
            if predicted_index == pred_index:
                if max_iou < predicted_iou:
                    max_iou_v_index = v_index
                    max_iou = predicted_iou

        if max_iou_v_index == -1:
            continue

        # удаляем все дубликаты этого предсказания
        for v, v_index in zip(intersections, range(len(intersections))):
            if v_index != max_iou_v_index:
                filtered_list = []
                for item in v[1]:
                    if item[2] != pred_index:
                        filtered_list.append(item)
                v[1] = filtered_list
    
    stats = defaultdict(lambda:[0, 0, 0])
    # все оставшиеся предсказания, кроме первого, считаются дубликатами
    for v in intersections:
        n_intersects = len(v[1])
        # если пересечения нет совсем, то ложноотрицательный
        if n_intersects == 0:
            stats[v[0]][2] += 1
            continue

        # все дубликаты воспринимаются как ложноположительные 
        stats[v[0]][1] += n_intersects - 1
        # если пересечение маленькое, то ложноположительный
        if v[1][0][1] < 0.5:
            stats[v[0]][1] += 1
        # если класс совпал, то истиноположительный
        elif v[1][0][0] == v[0]:
            stats[v[0]][0] += 1
        # если класс предсказан неверно, то ложноотрицательный
        else:
            stats[v[0]][2] += 1

    return stats


def recall(tp, fn):
    return tp / (tp + fn)


def percision(tp, fp):
    return tp / (tp + fp)


def config_matrix(path):
    out = []

    file = open(path)
    for line in file.readlines():
        out.append([])
        words = line.split()
        for word in words:
            out[-1].append(float(word))
    file.close()
    return np.array(out)

def print_percision_recall(main_dir, filename):
    ground = config_matrix(os.path.join(main_dir, 'output/YOLO_darknet', filename))
    predict = config_matrix(os.path.join(main_dir, 'predicts', filename))
    
    counts = confusion_counts(predict, ground)
    for k, v in counts.items():
        print('Class:', k)
        print('Percision:', percision(v[0], v[1]))
        print('Recall:', recall(v[0], v[2]))


#print_percision_recall('C:/Users/smirn/Documents/RSM/OpenLabelingCustom/main', 'file1_mp4_15283.txt')