import pandas as pd
import numpy as np

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
    files = os.listdir(root_dir)
    win_sizes = []
    for fl in files:
        size = fl.split('_')[1]
        win_sizes.append(size)
    files = np.asarray(files).reshape((-1, 1))
    win_sizes = np.asarray(win_sizes).reshape((-1, 1))
    return np.append(files, win_sizes, axis=1)


def read_data():
    name = (', score, sss, ggg, ttt\n'    '0,	0.066044398, 1,	1, 6 \n'    '1,	0.066044398, 1,	1, 6\n'    '2,	0.066044398, 1,	1, 6')
    print(name)
    #for name in files_name:
    data = pd.read_csv(StringIO(name))
    #print(data)
        	










if __name__ == "__main__":
    read_data()