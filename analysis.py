import pandas as pd
import numpy as np
from collections import OrderedDict

import os
from io import StringIO
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

def roc(labels, scores, true_subject, true_gesture, saveto=None):
    """Compute ROC curve and ROC area for each class"""
    gt_labels = label_convert(labels, true_subject, true_gesture)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    scores = torch.tensor(scores)
    # True/False Positive Rates.
    fpr, tpr, _ = roc_curve(gt_labels, scores)
    #logger.info(f'fpr, tpr, {fpr} {tpr}')
    roc_auc = auc(fpr, tpr)

    # Equal Error Rate
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    if saveto:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.2f, EER = %0.2f)' % (roc_auc, eer))
        plt.plot([eer], [1-eer], marker='o', markersize=5, color="navy")
        plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(saveto, "ROC.pdf"))
        plt.close()

    return roc_auc 


def read_files_name(root_dir):
    fnames = os.listdir(root_dir)
    win_sizes = []
    for fname in fnames:
        size = fname.split('_')[1]
        win_sizes.append(size)
    files = np.asarray(fnames).reshape((-1, 1))
    win_sizes = np.asarray(win_sizes).reshape((-1, 1))
    fname_winsize = np.append(files, win_sizes, axis=1)
    return fname_winsize

def read_data(root_dir, fname_winsize):
    performance = []
    for name, ws in fname_winsize:
       
        data = pd.read_csv(root_dir+name,index_col=[0])
        scores = data['scores']
        labels = data['labels']
        fpr, tpr, thresholds = roc_curve(labels, scores)

        fpr = np.array(fpr).reshape((-1,1))
        tpr = np.array(tpr).reshape((-1,1))
        thresholds = np.array(thresholds).reshape((-1,1))
        rate_triple = np.append(fpr,tpr,axis=1)
        rate_triple = np.append(rate_triple,thresholds,axis=1)
        
        roc_auc = auc(fpr, tpr)
        rate_triple = OrderedDict([
            ('winsize', ws), 
            ('rate_triple', rate_triple), 
            ('auc', roc_auc)])
        performance.append(rate_triple)
    return performance
        	
def find_threshold(labels, scores):
    fpr = dict()
    tpr = dict()
    thresholds = dict()
    roc_auc = dict()

    # labels = labels.cpu()
    # scores = scores.cpu()
    # True/False Positive Rates.
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    print(fpr[:10])
    print('~~~~~~~~~~~~~~~~')
    print(tpr[:10])
    print('~~~~~~~~~~~~~~~~')
    print(thresholds[:10])
    print('~~~~~~~~~~~~~~~~')
    print(roc_auc)
    print('~~~~~~~~~~~~~~~~')



if __name__ == "__main__":
    fnames = read_files_name('results_avg_img1/')
    print(fnames)
    thisdict = read_data('results_avg_img1/',fnames)
    print(len(thisdict))
    for row in thisdict:
        print(row['auc'])