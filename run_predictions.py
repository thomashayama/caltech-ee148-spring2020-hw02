import os
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
from numba import njit
from tqdm import tqdm

# Normalizes single channel
'''@njit()
def norm_im(im):
    new_im = im.copy()
    mean = np.mean(im[:, :])
    std = np.std(im[:, :])
    new_im = new_im - mean
    if std != 0:
        new_im[:, :] = new_im[:, :]/std
    norm = np.linalg.norm(new_im)
    if norm != 0:
        new_im = new_im/norm
    return new_im'''

#@njit()
def norm_im(im):
    new_im = im.copy()
    #for i in range(3):
        #mean = 0#np.mean(im[:, :, i])
        #std = np.std(im[:, :, i])
        #norm = np.linalg.norm(im[:, :, i])
        #new_im[:, :, i] = (new_im[:,:,i]-mean)/norm
    new_im = new_im / np.linalg.norm(new_im)
    return new_im

#@njit()
def compute_convolution(I, T, stride=1):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays)
    and returns a heatmap where each grid represents the output produced by
    convolution at each location. You can add optional parameters (e.g. stride,
    window_size, padding) to create additional functionality.
    '''
    (n_rows, n_cols, n_channels) = np.shape(I)
    max_score = np.linalg.norm(T)
    '''
    BEGIN YOUR CODE
    '''
    kernel = T.shape
    heatmap = np.zeros((int((n_rows-kernel[0])/stride)+1, int((n_cols-kernel[1])/stride)+1))
    for row in range(int((n_rows-kernel[0])/stride)+1):
        for col in range(int((n_cols-kernel[1])/stride)+1):
            im_section = norm_im(I[row*stride:row*stride+kernel[0], col*stride:col*stride+kernel[1]].astype(np.float64))
            #for i in range(n_channels):
            heatmap[row, col] += np.sum(im_section * T)
            heatmap[row, col] /= max_score
    '''
    END YOUR CODE
    '''
    #print(np.max(heatmap))
    return heatmap


def predict_boxes(heatmap):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    output = []

    '''
    BEGIN YOUR CODE
    '''

    '''
    As an example, here's code that generates between 1 and 5 random boxes
    of fixed size and returns the results in the proper format.
    '''

    threshold = .15
    box_height = len(avg_red)
    box_width = len(avg_red[0])
    (n_rows, n_cols) = np.shape(heatmap)

    for row in range(n_rows):
        for col in range(n_cols):
            if heatmap[row, col] >= threshold:
                output.append([row, col, row+box_width, col+box_height, heatmap[row, col]])

    '''
    END YOUR CODE
    '''

    return output


def detect_red_light_mf(I, T):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>.
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>.
    The first four entries are four integers specifying a bounding box
    (the row and column index of the top left corner and the row and column
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1.

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    '''
    BEGIN YOUR CODE
    '''

    # You may use multiple stages and combine the results

    bounding_boxes = []

    roi = []
    visited = set()

    r_threshold = 250

    def explore(row, col, id, visited, roi, r_threshold):
        queue = [[row, col]]
        while queue != []:
            row, col = queue.pop(0)
            if row >= 0 and col >= 0 and row < len(I[:, 0, 0]) and col < len(
                    I[0, :, 0]):
                if (row, col) not in visited:
                    visited.add((row, col))
                    if I[row, col, 0] >= r_threshold:
                        if row < roi[id][1]:
                            roi[id][1] = row
                        elif row > roi[id][3]:
                            roi[id][3] = row + 2
                        if col < roi[id][0]:
                            roi[id][0] = col
                        elif col > roi[id][2]:
                            roi[id][2] = col + 2
                        queue.append([row - 1, col])
                        queue.append([row + 1, col])
                        queue.append([row, col + 1])
                        queue.append([row, col - 1])

    id = 0
    for row in range(len(I[:, 0, 0])):
        for col in range(len(I[0, :, 0])):
            if (row, col) not in visited:
                visited.add((row, col))
                if I[row, col, 0] >= r_threshold:
                    roi.append([col, row, col + 2, row + 2])
                    explore(row - 1, col, id, visited, roi, r_threshold)
                    explore(row + 1, col, id, visited, roi, r_threshold)
                    explore(row, col + 1, id, visited, roi, r_threshold)
                    explore(row, col - 1, id, visited, roi, r_threshold)
                    id += 1

    def clipx(x):
        return int(max(0, min(len(I[0]), x)))

    def clipy(y):
        return int(max(0, min(len(I), y)))

    # Convolutions
    inflations = range(-2, 1)
    inflation_factor = 1.15
    zoomout = 2.0
    d_coef = .5
    temp = 1.92
    for label in roi:
        dy = label[3] - label[1]
        y_center = int((label[3] + label[1]) / 2)
        dx = label[2] - label[0]
        x_center = int((label[2] + label[0]) / 2)
        d_orig = max(dx, dy) * d_coef
        best_box = [0, 0, 0, 0, 0]
        for inflation in inflations:
            d = d_orig * zoomout
            new_im = I[clipy(y_center - d):clipy(y_center + d),
                     clipx(x_center - d):clipx(x_center + d)]
            new_im = np.asarray(Image.fromarray(new_im).resize(
                (int(len(T) * zoomout * inflation_factor**inflation),
                 int(len(T[0]) * zoomout * inflation_factor**inflation))))

            scale = (2*d/int(len(T) * zoomout * inflation_factor**inflation),
                    2*d/int(len(T[0]) * zoomout * inflation_factor**inflation))
            #print(scale)
            heatmap = compute_convolution(new_im, T)
            #boxes = predict_boxes(heatmap)
            best = np.argmax(heatmap)
            best = (int(best/len(heatmap[0])), best%len(heatmap[0]), np.max(heatmap))
            if best[2] > best_box[4]:
                best_box = [clipy(y_center - d + (best[1])*scale[1]), clipx(x_center - d + (best[0])*scale[0]),
                            clipy(y_center - d + (best[1] + 12)*scale[1]), clipx(x_center - d + (best[0] + 12)*scale[0]), best[2]]
            if best[2] > temp:
                print(best[2])
                fig = plt.figure()
                fig.add_subplot(1, 2, 1)
                plt.imshow(new_im)
                new_im = I[clipy(y_center - d + best[1]*scale[1]):clipy(y_center - d + (best[1] + 12)*scale[0]),
                     clipx(x_center - d + best[0]*scale[1]):clipx(x_center - d + (best[0] + 12)*scale[1])]
                fig.add_subplot(1, 2, 2)
                plt.imshow(new_im)
                plt.show()
        if best_box[4] > 0:
            bounding_boxes.append(best_box)
    #heatmap = compute_convolution(I, T)
    #output = predict_boxes(heatmap)

    '''
    END YOUR CODE
    '''

    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 5
        assert (bounding_boxes[i][4] >= 0.0) and (bounding_boxes[i][4] <= 1.0)

    return bounding_boxes


if __name__ == "__main__":
    # Note that you are not allowed to use test data for training.
    # set the path to the downloaded data:
    data_path = '../data/RedLights2011_Medium'

    # load splits:
    split_path = '../data/hw02_splits'
    file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
    file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

    # set a path for saving predictions:
    preds_path = '../data/hw02_preds'
    os.makedirs(preds_path, exist_ok=True) # create directory if needed

    # Set this parameter to True when you're done with algorithm development:
    done_tweaking = True

    '''
    Make predictions on the training set.
    '''
    avg_im = Image.open("avg_red.png")
    avg_red = norm_im(np.asarray(avg_im).astype(np.float64))

    preds_train = {}
    for i in tqdm(range(len(file_names_train))):
        # read image using PIL:
        I = Image.open(os.path.join(data_path, file_names_train[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_train[file_names_train[i]] = detect_red_light_mf(I, avg_red)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path, 'preds_train.json'), 'w') as f:
        json.dump(preds_train, f)

    if done_tweaking:
        '''
        Make predictions on the test set. 
        '''
        preds_test = {}
        for i in tqdm(range(len(file_names_test))):

            # read image using PIL:
            I = Image.open(os.path.join(data_path,file_names_test[i]))

            # convert to numpy array:
            I = np.asarray(I)

            preds_test[file_names_test[i]] = detect_red_light_mf(I, avg_red)

        # save preds (overwrites any previous predictions!)
        with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
            json.dump(preds_test,f)
