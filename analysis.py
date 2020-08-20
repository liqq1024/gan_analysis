import pandas as pd
import numpy as np
from collections import OrderedDict
import collections
import os
from io import StringIO
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score, accuracy_score, confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt

def files_name_wsize(root_dir):
    fnames = os.listdir(root_dir)
    win_sizes = []
    for fname in fnames:
        size = fname.split('_')[1]
        win_sizes.append(size)
    files = np.asarray(fnames).reshape((-1, 1))
    win_sizes = np.asarray(win_sizes).reshape((-1, 1))
    fname_winsize = np.append(files, win_sizes, axis=1)
    return fname_winsize

def read_data(root_dir):
    fnames = files_name_wsize(root_dir)
    performance = []
    for name, ws in fnames:
        data = pd.read_csv(root_dir+name,index_col=[0])
        scores = data['scores']
        labels = data['labels']
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        ls =[int(ws), roc_auc,  np.array(labels), np.array(scores), np.array(fpr), np.array(tpr)]
        performance.append(ls)
    return performance
        	

def draw_auc(stk_imgs, avg_imgs):
    winsize = [row[0] for row in stk_imgs]
    stk_imgs_auc = [row[1] for row in stk_imgs]
    avg_imgs_auc = [row[1] for row in avg_imgs]
    data = {'winsize': winsize, 'stk_imgs_auc': stk_imgs_auc, 'avg_imgs_auc': avg_imgs_auc}
    df = pd.DataFrame(data)
    df = df.sort_values(['winsize'])
    #df = df.cumsum()
    ax = df.plot(x='winsize')
    ax.set_title('Performance of AUC over window size')
    ax.set_xlabel('Window size')
    ax.set_ylabel('Max AUC')
    
    #ax.set_xticks(df['winsize'])
    fn = 'performance.png'
    print(fn)
    plt.savefig(fn)

def plt_roc(fpr, tpr, auc, ws):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(f'stack_roc_{ws}.png')


def majority_voting(labels, scores, win_size, mv_num):
    trial_size = 1000-win_size+1
    labels = labels.reshape((-1,trial_size))
    #print(labels)
    scores = scores.reshape((-1,trial_size))
    mv_labels = labels[:,0:mv_num ].reshape(-1,1)
    mv_scores = scores[:,0:mv_num].reshape(-1,1)
    fpr, tpr, thresholds = roc_curve(mv_labels, mv_scores)
    fpr = np.array(fpr).reshape((-1,1))
    tpr = np.array(tpr).reshape((-1,1))
    thresholds = np.array(thresholds).reshape((-1,1))
    Youden_index = tpr-fpr
    #Youden_index = (1-tpr)*fpr
    threshold =thresholds[np.argmax(Youden_index)] 
    print('*'*10)
    print(fpr[np.argmax(Youden_index)])
    print(tpr[np.argmax(Youden_index)])
    print('*'*10)
    pred_labels = np.where(mv_scores > threshold, 0, 1).reshape((-1,mv_num))
    print('threshold',threshold)
    print('pred_labels ',pred_labels)
    print('mv_labels',mv_labels.reshape(-1,mv_num))
    confu_metric1 = confusion_matrix(mv_labels,pred_labels.reshape((-1,1)))
    print('conf1 ', confu_metric1)
    trial_lb, trial_plb = trial_pred(mv_labels.reshape((-1,mv_num)),pred_labels, mv_num)
    # print(trial_lb)
    # print(trial_plb)
    confu_metric = []
    acc1 = accuracy_score(trial_lb,trial_plb)
    print(acc1)
    confu_metric = confusion_matrix(trial_lb,trial_plb)
    # print(confu_metric[0][0])
    # print(confu_metric[0][1])
    print('confu ',confu_metric)


def trial_pred(labels,pred_labels, mv_num):
    trial_lb = []
    trial_plb = []
    for i in range(0, pred_labels.shape[0]):
        p_count = np.count_nonzero(pred_labels[i][:] == 1)
        if p_count<=(mv_num//2):
            plb = 0
        else:
            plb = 1
        trial_plb.append(plb)
        if np.all(labels[i][:]==0):
            lb = 0
        elif np.all(labels[i][:]==1):
            lb = 1
        else:
            raise ValueError("the trial contains differnet labels")
        trial_lb.append(lb)

    return trial_lb, trial_plb

   
    

if __name__ == "__main__":
 
    performance_avg = read_data('res_avg/')
    performance_stack = read_data('res_stack/')
    draw_auc(performance_stack, performance_avg)
    for i, row in enumerate(performance_stack):
        scores = row[3]
        labels = row[2]
        ws = row[0]
        fpr = row[4]
        tpr = row[5]
        auc = row[1]
        majority_voting(labels, scores, ws, 591)
        plt_roc(fpr, tpr, auc, ws)
        