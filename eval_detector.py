import os
import json
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from tqdm import tqdm


def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    area_1 = (box_1[0] - box_1[2]) * (box_1[1] - box_1[3])
    area_2 = (box_2[0] - box_2[2]) * (box_2[1] - box_2[3])
    intersection = max(0, min(box_1[2], box_2[2]) - max(box_1[0], box_2[0])) * \
                   max(0, min(box_1[3], box_2[3]) - max(box_1[1], box_2[1]))
    iou = intersection / (area_1 + area_2 - intersection)
    
    assert (iou >= 0) and (iou <= 1.0)

    return iou


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.) 
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives. 
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0

    '''
    BEGIN YOUR CODE
    '''
    for pred_file, pred in preds.items():
        gt = gts[pred_file]
        pred.sort(key=lambda x: -x[4])
        for i in range(len(gt)):
            FN += 1
            for j in range(len(pred)):
                iou = compute_iou(pred[j][:4], gt[i])
                if pred[j][4] < conf_thr:
                    break
                elif i == 0:
                    FP += 1

                if iou > iou_thr:
                    TP += 1
                    FN -= 1
                    break
    FP -= TP

    #print(TP, FP, FN)

    '''
    END YOUR CODE
    '''

    return TP, FP, FN

# set a path for predictions and annotations:
preds_path = '../data/hw02_preds'
gts_path = '../data/hw02_splits'

# load splits:
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Load training data. 
'''
with open(os.path.join(preds_path,'preds_train.json'), 'r') as f:
    preds_train = json.load(f)

#for key, val in preds_train.items():
#    print(key)
#    print(val)
#    exit()
    
with open(os.path.join(gts_path, 'annotations_train.json'), 'r') as f:
    gts_train = json.load(f)

# For a fixed IoU threshold, vary the confidence thresholds.
# The code below gives an example on the training set for one IoU threshold. 

iou_thrs = [0.25, 0.5, .75]
'''
confidence_thrs = np.array([])
for fname in preds_train:
    confidence_thrs = np.append(confidence_thrs, np.array([bbox[4] for bbox in preds_train[fname]],dtype=float)) # using (ascending) list of confidence scores as thresholds
confidence_thrs = np.sort(np.unique(confidence_thrs[::10]))
#print(len(confidence_thrs))

tp_train = np.zeros(len(confidence_thrs))
fp_train = np.zeros(len(confidence_thrs))
fn_train = np.zeros(len(confidence_thrs))

for iou_thr in iou_thrs:
    for i, conf_thr in tqdm(enumerate(confidence_thrs)):
        tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_train, gts_train, iou_thr=iou_thr, conf_thr=conf_thr)

    N = 0
    for val in gts_train.values():
        N += len(val)
    # Plot training set PR curves
    precision = np.zeros(len(confidence_thrs))
    for i in range(len(confidence_thrs)):
        if tp_train[i] + fp_train[i] == 0:
            precision[i] = 1
        else:
            precision[i] = tp_train[i] / (tp_train[i] + fp_train[i])

    recall = tp_train / N
    precision = np.insert(precision, 0, 0)
    recall = np.insert(recall, 0, 1.0)
    precision = np.append(precision, 1.0)
    recall = np.append(recall, 0)
    plt.figure()
    plt.plot(recall, precision)
    plt.title("Precision Recall Curve for Train Set IOU > " + str(iou_thr))
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.show()
'''
if done_tweaking:
    print('Code for plotting test set PR curves.')

    with open(os.path.join(preds_path, 'preds_test_weak.json'), 'r') as f:
        preds_test = json.load(f)

    with open(os.path.join(gts_path, 'annotations_test.json'), 'r') as f:
        gts_test = json.load(f)

    confidence_thrs = np.array([])
    for fname in preds_test:
        confidence_thrs = np.append(confidence_thrs, np.array(
            [bbox[4] for bbox in preds_test[fname]],
            dtype=float))  # using (ascending) list of confidence scores as thresholds
    confidence_thrs = np.sort(np.unique(confidence_thrs[::2]))
    # print(len(confidence_thrs))

    tp_test = np.zeros(len(confidence_thrs))
    fp_test = np.zeros(len(confidence_thrs))
    fn_test = np.zeros(len(confidence_thrs))

    for iou_thr in iou_thrs:
        for i, conf_thr in tqdm(enumerate(confidence_thrs)):
            tp_test[i], fp_test[i], fn_test[i] = compute_counts(preds_test,
                                                                   gts_test,
                                                                   iou_thr=iou_thr,
                                                                   conf_thr=conf_thr)

        N = 0
        for val in gts_test.values():
            N += len(val)
        # Plot training set PR curves
        precision = np.zeros(len(confidence_thrs))
        for i in range(len(confidence_thrs)):
            if tp_test[i] + fp_test[i] == 0:
                precision[i] = 1
            else:
                precision[i] = tp_test[i] / (tp_test[i] + fp_test[i])

        recall = tp_test / N
        precision = np.insert(precision, 0, 0)
        recall = np.insert(recall, 0, 1.0)
        precision = np.append(precision, 1.0)
        recall = np.append(recall, 0)
        plt.figure()
        plt.plot(recall, precision)
        plt.title("Precision Recall Curve for Test Set IOU > " + str(iou_thr))
        plt.ylabel("Precision")
        plt.xlabel("Recall")
        plt.show()