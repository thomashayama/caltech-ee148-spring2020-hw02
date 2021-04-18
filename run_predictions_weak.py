import os
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
from numba import njit
from tqdm import tqdm

# Normalizes
#@njit()
def norm_im(im):
    new_im = im.copy()
    new_im = new_im/np.linalg.norm(im)
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
    max_score = 1
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

    threshold = .9
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


def detect_red_light_mf(I):
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
    T = avg_red

    heatmap = compute_convolution(I, T)
    output = predict_boxes(heatmap)

    '''
    END YOUR CODE
    '''

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output

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
        I = Image.open(os.path.join(data_path,file_names_train[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_train[file_names_train[i]] = detect_red_light_mf(I)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path, 'preds_train_weak.json'), 'w') as f:
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

            preds_test[file_names_test[i]] = detect_red_light_mf(I)

        # save preds (overwrites any previous predictions!)
        with open(os.path.join(preds_path,'preds_test_weak.json'),'w') as f:
            json.dump(preds_test,f)
