from PIL import Image, ImageDraw
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from run_predictions import compute_convolution, norm_im

def label_image(im, preds, width=2):
    draw = ImageDraw.Draw(im)
    for pred in preds:
        draw.rectangle([int(pred[3]), int(pred[2]), int(pred[1]),
                        int(pred[0])], outline='red', width=width)
    return im

if __name__=="__main__":

    # set the path to the downloaded data:
    data_path = '../data/RedLights2011_Medium'

    # set a path for saving predictions:
    annotations_path = '../data/hw02_preds/preds_experimental.json'
    #annotations_path = '../data/hw02_splits/annotations_train.json'

    # get sorted list of files:
    file_names = sorted(os.listdir(data_path))

    # remove any non-JPEG files:
    file_names = [f for f in file_names if '.jpg' in f]

    # get template
    avg_im = Image.open("avg_red.png")
    T = norm_im(np.asarray(avg_im).astype(np.float64))

    # load preds
    with open(annotations_path, 'r') as f:
        preds = json.load(f)

    for file, pred in preds.items():
        im = Image.open(os.path.join(data_path,file))
        im_np = np.asarray(im)
        plt.figure()
        plt.imshow(label_image(im, pred))
        plt.show()
        plt.figure()
        plt.imshow(compute_convolution(im_np, T))
        plt.show()

