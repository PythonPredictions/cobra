"""
======================================================================================
--------------------------------------- Evaluation Class code ------------------------
======================================================================================
author: jan.benisek@pythonpredictins.com - benoit.vandekerkhove@pythonpredictions.com
date: 23/09/2019
purpose: library for model evaluation class

"""
#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as mt
from typing import Tuple
#%%


class Evaluator():
    ''' 
    Class to evaluate models

    Parameters
    -----------
    y_true : array, shape = [1, n_features]
        array with true values

    y_pred_p : array, shape = [1, n_features]
        array with predicted values (probabilities)

    lift_at : int , default=0.05 
        calculate lift at given level (0-1)

    save_pth : str, default=None
        path to where save the plot

    binary_cutoff : float, default=0.5
        cutoff to convert predictions to binary

    '''
    
    def __init__(self, y_true: np.ndarray, y_pred_p: np.ndarray, 
                lift_at: float=0.05, save_pth: str=None, binary_cutoff: int=0.5):
        
        self.y_true = y_true.flatten()
        self.y_pred_p = y_pred_p.flatten() #As probability
        self.lift_at = lift_at
        self.save_pth = save_pth
        self.binary_cutoff = binary_cutoff

        self.y_pred_b = np.where(self.y_pred_p > self.binary_cutoff,1,0)




    '''=============================================================
    ----------------------------- PLOTS ----------------------------
    ============================================================='''
    def plotROCCurve(self, desc: str=None):
        '''
        Plot ROC curve and print best cutoff value
        Transform probabilities predictions to bool based on best AUC based cutoff
        
        Parameters
        ----------
        desc : str, default=None
            description of the plot, used also as a name of saved plot

        '''   
        if desc is None:
            desc = ''
            
        fpr,tpr,thresholds = mt.roc_curve(self.y_true,self.y_pred_p)
        
        #---------------------------
        #Calculate AUC
        #--------------------------
        score = mt.roc_auc_score(self.y_true, self.y_pred_p)
    
        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(fpr,tpr, color='darkorange', lw=2, label='ROC curve (area = {s:.3})'.format(s=score))
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlabel('False Positive Rate', fontsize=15)
        ax.set_ylabel('True Positive Rate', fontsize=15)
        ax.legend(loc="lower right")
        ax.set_title('ROC Curve {}' .format(desc), fontsize=20)
            
        if save_pth is not None:
            plt.savefig(save_pth, format='png', dpi=300, bbox_inches='tight')
            
        plt.show()

    '''=============================================================
    ---------------------------- METRICS ---------------------------
    ============================================================='''
            
    def printPerformance(self):
        '''
        Print out performance measures

        EV.printPerformance()
        %timeit 2min 19s ± 784 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        '''
        
        if self.threshold != np.nan :
            out_perfo = self._evaluation()
        
            print('=== Test on', self.test_on, '===')
            print('Precision: {s:.3}'.format(s=out_perfo['precision'])) #If we mark customer as a churner, how often we are correct
            print('Accuracy: {s:.3}'.format(s=out_perfo['accuracy'])) #Overall performance
            print('Recall: {s:.3}'.format(s=out_perfo['recall'])) #How many churners can the model detect
            print('F1 Score: {s:.3}'.format(s=out_perfo['F1'])) # 2 * (precision * recall) / (precision + recall)
            print('Lift at top {l}%: {s:.3}'.format(l=self.lift_at*100, s=out_perfo['lift'])) # 2 * (precision * recall) / (precision + recall)
            print('AUC: {s:.3}'.format(s=out_perfo['AUC'])) # 2 * (precision * recall) / (precision + recall)
        
        else :
            raise ValueError('Please call .plotROCCurve() method first to get the best threshold for probabilities, and try again')
       
    def plotLift(self, desc : str=None, save_pth : str=None):
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
        lifts = [Evaluator.liftCalculator(y_true=self.y_true, y_pred=self.y_pred_p, lift_at=perc_lift)
                for perc_lift in np.arange(0.05,1.05,0.05)]
        
        #---------------------
        #------- PLOT --------
        #---------------------
        if desc is None:
                desc = ''
        
        fig, ax = plt.subplots(figsize=(8,5))
        plt.style.use('seaborn-darkgrid')
        
        nrows = len(lifts)
        x_labels = [nrows/2-x/2 for x in np.arange(0,nrows,1)]
    
        #plt.bar(x_labels[::-1], df['lift'].values.tolist(), align='center', color="cornflowerblue")
        plt.bar(x_labels[::-1], lifts, align='center', color="green", width=0.2)
        plt.ylabel('lift', fontsize=15)
        plt.xlabel('decile', fontsize=15)
        ax.set_xticks(x_labels)
        ax.set_xticklabels(x_labels)
        
        plt.axhline(y=1, color='darkorange', linestyle='--', 
                    xmin=0.05, xmax=0.9, linewidth=3, label='Baseline')
    
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


            
    '''-------------------------------------------------------------------
    -------------------------------- UTILS -------------------------------
    -------------------------------------------------------------------'''
    def estimateCutoff(self) -> float:
        '''
        Estimates optimal cutoff based on maximization of AUC curve
        https://stackoverflow.com/questions/28719067/roc-curve-and-cut-off-point-python

        Parameters
        ----------
        None

        Returns
        -------
        best_cutoff : float
            optimal cutoff as a float <0;1>

        '''
        fpr,tpr,thresholds = mt.roc_curve(self.y_true,self.y_pred_p)
        i = np.arange(len(tpr)) 
        roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 
                            'threshold' : pd.Series(thresholds, index=i)})
        roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
        
        best_cutoff =  list(roc_t['threshold'])

        return best_cutoff[0]
        
    
    def _testA(self, test :  np.ndarray, pred : np.ndarray, train_M : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ''' 
        Limits the evaluation to potential A offers 
        (that a customer has not purchase in the train timeframe)
        
        Parameters
        ----------
        test: true values -> array
        pred: predictions as probabilities -> array
        train_M : train matrix of interactions -> ndarray
        
        Output
        ------
        testA: vector of interaction on potential A offers -> array
        predA: vector of predictions on potential A offers -> array
        '''
    
        train = train_M.flatten()
        testA = np.where(train>0, np.nan, test)
        predA = np.where(train>0, np.nan, pred)
        testA = testA[testA>=0]
        predA = predA[predA>=0]
        
        return testA, predA
    
    def _evaluation(self):
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

        dict_perfo = {'precision': mt.precision_score(self.y_true, self.y_pred_b),
                      'accuracy': mt.accuracy_score(self.y_true, self.y_pred_b),
                      'recall': mt.recall_score(self.y_true, self.y_pred_b),
                      'F1': mt.f1_score(self.y_true, self.y_pred_b, average=None)[1],
                      'lift': np.round(Evaluator.liftCalculator(y_true=self.y_true,
                                                                y_pred=self.y_pred_p,
                                                                lift_at=self.lift_at),2),
                      'AUC': mt.roc_auc_score(self.y_true, self.y_pred_p)
                      }
        return dict_perfo
    
    @staticmethod
    def liftCalculator(y_true : np.ndarray, y_pred : np.ndarray, lift_at : float=0.05, **kwargs) -> float:
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
        
        50.3 µs ± 1.94 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
        ''' 
        #Make sure it is numpy array
        y_true_ = np.array(y_true)
        y_pred_ = np.array(y_pred)
        
        #Make sure it has correct shape
        y_true_ = y_true_.reshape(len(y_true_),1)
        y_pred_ = y_pred_.reshape(len(y_pred_),1)
        
        #Merge data together
        y_data = np.hstack([y_true_, y_pred_])
        
        #Calculate necessary variables
        nrows = len(y_data)
        stop = int(np.floor(nrows*lift_at))
        avg_incidence = np.einsum('ij->j',y_true_)/float(len(y_true_))
        
        #Sort and filter data
        data_sorted = y_data[y_data[:,1].argsort()[::-1]][:stop,0].reshape(stop, 1)
        
        #Calculate lift (einsum is very fast way of summing, needs specific shape)
        inc_in_top_n = np.einsum('ij->j',data_sorted)/float(len(data_sorted))
        
        lift = np.round(inc_in_top_n/avg_incidence,2)[0]
        
        return lift

    '''-------------------------------------------------------------------
    ------------------------JUST IN CASE -------------------------------
    -------------------------------------------------------------------'''
    
    def plotConfusionMatrix(self, labels : list=None, color : str='Reds', 
                            save_pth : str=None, desc : str=None):
        '''
        Plot Confusion matrix 
        
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
            
        cm = mt.confusion_matrix(self.y_true, self.y_pred_b)
        
        fig, ax = plt.subplots(figsize=(8,5))
        ax = sns.heatmap(cm, annot=cm.astype(str), fmt="s", cmap=color, xticklabels=labels, yticklabels=labels)
        ax.set_title('Confusion matrix {}'.format(desc), fontsize=20)
        
        if save_pth is not None:
            plt.savefig(save_pth, format='png', dpi=300, bbox_inches='tight')
            
        plt.show()   
        
    def plotCumulativeGains(self, save_pth : str=None, desc : str=None):
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
        df_y_pred = pd.DataFrame({"y":self.y_true, "y_pred":self.y_pred_p}).sort_values(by='y_pred', ascending=False).reset_index(drop=True)
        cgains = [0]    
        for stop in (np.linspace(0.01,1,100)*nrows).astype(int):
            cgains.append(round(df_y_pred.loc[:stop,'y'].sum()/npositives*max(100,1),2))
          
        #---------------------------
        #Plot it
        #---------------------------        
        plt.style.use('seaborn-darkgrid')
        fig, ax_cgains = plt.subplots(figsize=(8,5))
        ax_cgains.plot(cgains, color='blue', linewidth=3, label='cumulative gains')    
        ax_cgains.plot(ax_cgains.get_xlim(), ax_cgains.get_ylim(), linewidth=3, ls="--", color="darkorange", label='random selection')
        ax_cgains.set_title('Cumulative Gains ' + desc, fontsize=20)
        
        ax_cgains.set_title('Cumulative Gains {}' .format(desc), fontsize=20)
        #Format axes
        ax_cgains.set_xlim([0,100])
        ax_cgains.set_ylim([0,100])
        #Format ticks
        ax_cgains.set_yticklabels(['{:3.0f}%'.format(x) for x in ax_cgains.get_yticks()])
        ax_cgains.set_xticklabels(['{:3.0f}%'.format(x) for x in ax_cgains.get_xticks()])
        #Legend
        ax_cgains.legend(loc='lower right')
            
        if save_pth is not None:
            plt.savefig(save_pth, format='png', dpi=300, bbox_inches='tight')
            
        plt.show()

    def plotCumulativeResponse(self, desc : str=None, save_pth : str=None):
        #---------------------
        #-- CALCULATE LIFT ---
        #---------------------
        inc_rate = self.y_true.mean()
        lifts = [Evaluator.liftCalculator(y_true=self.y_true, y_pred=self.y_pred_p, lift_at=perc_lift)
                for perc_lift in np.arange(0.1,1.1,0.1)]
        lifts = np.array(lifts)*inc_rate*100
        #---------------------
        #------- PLOT --------
        #---------------------
        if desc is None:
                desc = ''
        
        fig, ax = plt.subplots(figsize=(8,5))
        #plt.style.use('seaborn-darkgrid')
        plt.style.use('default')
        
        nrows = len(lifts)
        x_labels = [nrows-x for x in np.arange(0,nrows,1)]
    
        #plt.bar(x_labels[::-1], df['lift'].values.tolist(), align='center', color="cornflowerblue")
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

def plotIncidence(df, variable, dim=(12,8)):
    '''
    Method plots Incidence plot on train partition
    Returns plot
    ----------------------------------------------------
    df: dataframe with cleaned, binned, partitioned and prepared data
    variable: variable for which the incidence plot will be shown`
    dim: tuple with width and lentgh of the plot
    ---------------------------------------------------- 
    '''
    def masterOfOrder(x):     
        ''' 
        Function converts interval or string (category) to a number, so the incidence plot can be orderd.
        In case of interval -> '(151, 361]' to integer 151.
        In case of string -> order is alphabetical
        Missings and Non-significants are always put at the end
        
        Parameters
        ----------
        x: value to be converted
        
        Output
        ------
        Order of given value
        '''
        x_split = x.split(',')[0]
        replace_strings = (('...', '0'),('Missing','999999999999'), ('Non-significants','999999999999'))
        for repl_str in replace_strings:
                    x_split = x_split.replace(repl_str[0], repl_str[1])
        x_split = x_split.strip("()[]")
        
        try:
            order = float(x_split)
        except:
            LETTERS = {letter: index for index, letter in enumerate(ascii_lowercase, start=1)}
            order = LETTERS[x[0].lower()]
            
        return order
                
    plt.style.use('seaborn-darkgrid')
    
    #----------------------------------
    #------  Prepare the data  --------
    #----------------------------------
    #Set up the variable and dataframe
    var_prefix = 'B_' + variable
    df_plt = df[['TARGET', var_prefix]][df['PARTITION'] == 'train'].copy()
    
    #Aggregate the data
    avg_inc_rate = df_plt['TARGET'].mean()
    
    aggregations = {
                    'bin_inc_rate': 'mean',
                    'bin_size': 'count'
                    }
    df_plt = df_plt.groupby(var_prefix, as_index=False)['TARGET'].agg(aggregations)
    df_plt['avg_inc_rate'] = avg_inc_rate
    
    #create a sort column and sort by it    
    df_plt['sort_by'] = df_plt[var_prefix].apply(lambda x: masterOfOrder(x))
    df_plt.sort_values(by='sort_by', ascending=True, inplace=True)
    df_plt.reset_index(inplace=True)
    
    #----------------------------------
    #-----  Plot the incidence  -------
    #----------------------------------
    fig, ax = plt.subplots(figsize=dim)
    ##First Axis
    #Bin size
    y_pos = np.arange(len(df_plt[var_prefix]))
    plt.bar(y_pos, df_plt['bin_size'].values.tolist(), align='center', color="cornflowerblue")
    plt.xticks(y_pos, df_plt[var_prefix])
    plt.ylabel('Bin Size')
    plt.xlabel(variable + ' Bins')
    
    max_inc = max(df_plt['bin_inc_rate'])
    
    ##Second Axis
    ax2 = ax.twinx()
    #incidence rate per bin
    plt.plot(df_plt['bin_inc_rate'], color="darkorange", marker=".", markersize=20, linewidth=3, label='incidence rate per bin')
    plt.plot(df_plt['avg_inc_rate'], color="dimgrey", linewidth=4, label='average incidence rate')
    ax2.plot(np.nan, "cornflowerblue", linewidth=6, label = 'bin size') #dummy line to have label on second axis from first
    ax2.set_yticks(np.arange(0, max_inc+0.05, 0.05))
    ax2.set_yticklabels(['{:3.1f}%'.format(x*100) for x in ax2.get_yticks()])
    plt.ylabel('Incidence')
    
    ##Set Axis
    sns.despine(ax=ax, right=True, left=True)
    sns.despine(ax=ax2, left=True, right=False)
    ax2.spines['right'].set_color('white')
    
    #remove white line from second grid axes
    #the white lines are reguler, Spyder sometimes fails to visualize it (try to export the pic!)
    ax2.grid(False)
    
    ##Description
    fig.suptitle('Incidence Plot - ' + variable, fontsize=20, y=1.02)
    ax2.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=1, mode="expand", borderaxespad=0.)
    plt.show()
