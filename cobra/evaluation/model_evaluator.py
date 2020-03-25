"""
Created on Fri Apr 12 09:36:37 2019
@author: AP_JBENEK
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score


class Evaluator():

    def __init__(self, y_true, y_pred_p, threshold=0.5, lift_at=0.1):
        self.y_true = y_true
        self.y_pred_p = y_pred_p  # As probability
        self.lift_at = lift_at
        self.threshold = threshold

        #Convert to bool
        self.y_pred_b = np.array([0 if pred <= self.threshold else 1
                                  for pred in self.y_pred_p])

    def plotROCCurve(self, save_pth=None, desc=None):
        '''
        Plot ROC curve and print best cutoff value

        Parameters
        ----------
        y_true: True values of target y
        proba: Predicted values of target y, probabilities
        save: whether plot should be saved (if yes, then now shown)
        desc: description of the plot, used also as a name of saved plot
        '''
        if desc is None:
            desc = ''

        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_pred_p)

        #---------------------------
        #Calculate AUC
        #--------------------------
        out_perfo = self.evaluation()
        score = out_perfo['AUC']

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(fpr, tpr, color='darkorange', lw=2,
                label='ROC curve (area = {s:.3})'.format(s=score))
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlabel('False Positive Rate', fontsize=15)
        ax.set_ylabel('True Positive Rate', fontsize=15)
        ax.legend(loc="lower right")
        ax.set_title('ROC Curve {}' .format(desc), fontsize=20)

        if save_pth is not None:
            plt.savefig(save_pth, format='png', dpi=300, bbox_inches='tight')

        plt.show()

        #Best cutoff value
        #i want value where FPR is highest and FPR is lowest
        #https://stackoverflow.com/questions/28719067/roc-curve-and-cut-off-point-python
        i = np.arange(len(tpr))
        roc = pd.DataFrame({'tf': pd.Series(tpr-(1-fpr), index=i),
                            'threshold': pd.Series(thresholds, index=i)})
        roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

        best_cutoff =  list(roc_t['threshold'])
        print(f'Best cutoff value for probability is: {best_cutoff[0]}')

    def plotConfusionMatrix(self, labels=None, color='Reds', save_pth=None, desc=None):
        '''
        Plot Confusion matrix with performance measures

        Parameters
        ----------
        y_test: True values of target y
        pred: Predicted values of target y, boolean
        labels: labels for the matrix, if empty, values from y_test_ are used
        color: Color of the matrix, its a cmap, so many values possible
        save: whether plot should be saved (if yes, then now shown)
        desc: description of the plot, used also as a name of saved plot
        '''
        if labels is None:
            labels = [str(lab) for lab in np.unique(self.y_true)]

        if desc is None:
            desc = ''

        cm = confusion_matrix(self.y_true, self.y_pred_b)

        fig, ax = plt.subplots(figsize=(8,5))
        ax = sns.heatmap(cm, annot=cm.astype(str), fmt="s", cmap=color,
                         xticklabels=labels, yticklabels=labels)
        ax.set_title('Confusion matrix {}'.format(desc), fontsize=20)

        if save_pth is not None:
            plt.savefig(save_pth, format='png', dpi=300, bbox_inches='tight')

        plt.show()

        out_perfo = self.evaluation()

        # If we mark customer as a churner, how often we are correct
        print('Precision: {s:.3}'.format(s=out_perfo['precision']))
        # Overall performance
        print('Accuracy: {s:.3}'.format(s=out_perfo['accuracy']))
        # How many churners can the model detect
        print('Recall: {s:.3}'.format(s=out_perfo['recall']))
        # 2 * (precision * recall) / (precision + recall)
        print('F1 Score: {s:.3}'.format(s=out_perfo['F1']))
        # 2 * (precision * recall) / (precision + recall)
        print('Lift at top {l}%: {s:.3}'
              .format(l=self.lift_at*100, s=out_perfo['lift']))
        # 2 * (precision * recall) / (precision + recall)
        print('AUC: {s:.3}'.format(s=out_perfo['AUC']))

    def plotCumulativeGains(self, save_pth=None, desc=None):
        '''
        Functions plot cumulative gains

        Parameters
        ----------
        save: whether plot should be saved (if yes, then now shown)
        desc: description of the plot, used also as a name of saved plot
        '''
        if desc is None:
            desc = ''

        #---------------------------
        #Calculate cumulative gains
        #--------------------------
        nrows = len(self.y_true)
        npositives = self.y_true.sum()
        df_y_pred = (pd.DataFrame({"y": self.y_true, "y_pred": self.y_pred_p})
                     .sort_values(by='y_pred', ascending=False)
                     .reset_index(drop=True))
        cgains = [0]
        for stop in (np.linspace(0.01, 1, 100) * nrows).astype(int):
            cgains.append(round(df_y_pred.loc[:stop, 'y'].sum()/npositives*max(100, 1), 2))

        #---------------------------
        #Plot it
        #---------------------------
        plt.style.use('seaborn-darkgrid')
        fig, ax_cgains = plt.subplots(figsize=(8, 5))
        ax_cgains.plot(cgains, color='blue', linewidth=3,
                       label='cumulative gains')
        ax_cgains.plot(ax_cgains.get_xlim(), ax_cgains.get_ylim(), linewidth=3,
                       ls="--", color="darkorange", label='random selection')
        ax_cgains.set_title('Cumulative Gains ' + desc, fontsize=20)

        ax_cgains.set_title('Cumulative Gains {}' .format(desc), fontsize=20)
        #Format axes
        ax_cgains.set_xlim([0, 100])
        ax_cgains.set_ylim([0, 100])
        #Format ticks
        ax_cgains.set_yticklabels(['{:3.0f}%'.format(x)
                                   for x in ax_cgains.get_yticks()])
        ax_cgains.set_xticklabels(['{:3.0f}%'.format(x)
                                   for x in ax_cgains.get_xticks()])
        #Legend
        ax_cgains.legend(loc='lower right')

        if save_pth is not None:
            plt.savefig(save_pth, format='png', dpi=300, bbox_inches='tight')

        plt.show()

    def plotLift(self, desc=None, save_pth=None):
        '''
        Method plots lift per decile

        Parameters
        ----------
        save: whether plot should be saved (if yes, then now shown)
        desc: description of the plot, used also as a name of saved plot
        '''
        #---------------------
        #-- CALCULATE LIFT ---
        #---------------------
#        inc_rate = self.y_true.mean()
        lifts = [Evaluator.liftCalculator(y_true=self.y_true,
                                          y_pred=self.y_pred_p,
                                          lift_at=perc_lift)
                 for perc_lift in np.arange(0.1, 1.1, 0.1)]

        #---------------------
        #------- PLOT --------
        #---------------------
        if desc is None:
                desc = ''

        fig, ax = plt.subplots(figsize=(8,5))
        plt.style.use('seaborn-darkgrid')

        nrows = len(lifts)
        x_labels = [nrows-x for x in np.arange(0, nrows, 1)]

        plt.bar(x_labels[::-1], lifts, align='center', color="cornflowerblue")
        plt.ylabel('lift', fontsize=15)
        plt.xlabel('decile', fontsize=15)
        ax.set_xticks(x_labels)
        ax.set_xticklabels(x_labels)

        plt.axhline(y=1, color='darkorange', linestyle='--',
                    xmin=0.1, xmax=0.9, linewidth=3, label='Baseline')

        #Legend
        ax.legend(loc='upper right')

        ##Set Axis - make them pretty
        sns.despine(ax=ax, right=True, left=True)

        #Remove white lines from the second axis
        ax.grid(False)

        ##Description
        ax.set_title('Cumulative Lift {}'.format(desc), fontsize=20)

        if save_pth is not None:
            plt.savefig(save_pth, format='png', dpi=300, bbox_inches='tight')

        plt.show()

    def plotCumulativeResponse(self, desc=None, save_pth=None):
        #---------------------
        #-- CALCULATE LIFT ---
        #---------------------
        inc_rate = self.y_true.mean()
        lifts = [Evaluator.liftCalculator(y_true=self.y_true,
                                          y_pred=self.y_pred_p,
                                          lift_at=perc_lift)
                 for perc_lift in np.arange(0.1, 1.1, 0.1)]
        lifts = np.array(lifts)*inc_rate*100
        #---------------------
        #------- PLOT --------
        #---------------------
        if desc is None:
                desc = ''

        fig, ax = plt.subplots(figsize=(8, 5))
        #plt.style.use('seaborn-darkgrid')
        plt.style.use('default')

        nrows = len(lifts)
        x_labels = [nrows-x for x in np.arange(0, nrows, 1)]

        plt.bar(x_labels[::-1], lifts, align='center', color="#00ccff")
        plt.ylabel('response (%)', fontsize=16)
        plt.xlabel('decile', fontsize=16)
        ax.set_xticks(x_labels)
        ax.set_xticklabels(x_labels)

        plt.axhline(y=inc_rate*100, color='#ff9500', linestyle='--',
                    xmin=0.05, xmax=0.95, linewidth=3, label='Incidence')

        #Legend
        ax.legend(loc='upper right')

        ##Set Axis - make them pretty
        sns.despine(ax=ax, right=True, left=True)

        #Remove white lines from the second axis
        ax.grid(False)

        ##Description
        ax.set_title('Cumulative response {}'.format(desc), fontsize=20)

        if save_pth is not None:
            plt.savefig(save_pth, format='png', dpi=300, bbox_inches='tight')

        plt.show()

    def evaluation(self):
        '''
        Convenient function, returns various performance measures in a dict

        Parameters
        ----------
        y_true: true values
        y_pred: predictions as booleans

        Output
        ------
        Returns dictionary with the measures
        '''

        dict_perfo = {'precision': precision_score(self.y_true, self.y_pred_b),
                      'accuracy': accuracy_score(self.y_true, self.y_pred_b),
                      'recall': recall_score(self.y_true, self.y_pred_b),
                      'F1': f1_score(self.y_true, self.y_pred_b,
                                     average=None)[1],
                      'lift': np.round(Evaluator
                                       .liftCalculator(y_true=self.y_true,
                                                       y_pred=self.y_pred_p,
                                                       lift_at=self.lift_at),
                                       2),
                      'AUC': roc_auc_score(self.y_true, self.y_pred_p)
                      }
        return dict_perfo

    @staticmethod
    def liftCalculator(y_true, y_pred, lift_at=0.05, **kwargs):
        '''
        Calculates lift given two arrays on specified level

        Parameters
        ----------
        y_true: numpy array with true values
        y_pred: numpy array with predictions (probabilities)
        lift_at: lift at what top percentage

        Output
        ------
        Scalar value, lift.

        50.3 µs ± 1.94 µs per loop (mean ± std. dev. of 7 runs,
                                    10000 loops each)
        '''
        #Make sure it is numpy array
        y_true_ = np.array(y_true)
        y_pred_ = np.array(y_pred)

        #Make sure it has correct shape
        y_true_ = y_true_.reshape(len(y_true_), 1)
        y_pred_ = y_pred_.reshape(len(y_pred_), 1)

        #Merge data together
        y_data = np.hstack([y_true_, y_pred_])

        #Calculate necessary variables
        nrows = len(y_data)
        stop = int(np.floor(nrows*lift_at))
        avg_incidence = np.einsum('ij->j', y_true_)/float(len(y_true_))

        #Sort and filter data
        data_sorted = (y_data[y_data[:, 1].argsort()[::-1]][:stop, 0]
                       .reshape(stop, 1))

        #Calculate lift (einsum is very fast way of summing,
        # needs specific shape)
        inc_in_top_n = np.einsum('ij->j', data_sorted)/float(len(data_sorted))

        lift = np.round(inc_in_top_n/avg_incidence, 2)[0]

        return lift
