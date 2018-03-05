''' 
======================================================================================================================
--------------------------------------------------  UNIVARIATE SELECTION  --------------------------------------------
======================================================================================================================
'''
import numpy as np
import pandas as pd
from sklearn import metrics

class UnivariateSelection(object):
    '''
    Class for Univariate Selection.
    Calculates AUC and correlation matrix
    ----------------------------------------------------
    Author: Jan Benisek, Python Predictions
    Date: 19/02/2018
    ----------------------------------------------------
    ***PARAMETERS***
    :preselect_auc:             Minimal treshold for AUC selection
    :preselect_overtrain:       Threshold for difference between train and test performance
    ---------------------------------------------------- 
    '''
    
    def __init__(self, preselect_auc, preselect_overtrain):
        ''' ***PARAMETERS*** '''
        self.preselect_auc = preselect_auc
        self.preselect_overtrain = preselect_overtrain
        
    def fit(self, df):
        '''
        Method fits (=performs) Univariate selection
        Returns auc, correlation and list with filtered variables
        ----------------------------------------------------
        df: transformed dataset
        ---------------------------------------------------- 
        '''
        key_clmns = ["ID","TARGET","PARTITION"]
        
        ##AUC selection
        df_auc = self._calcFilterAUC(df, key_clmns)
        
        ##Correlation
        df_corr = self._calcCorr(df, key_clmns)
        
        return df_auc, df_corr
        
        
    def _calcFilterAUC(self, df, key_clmns):
        '''
        Method calculates AUC for train/test
        Returns DF with AUC higher than given threshold, drops overfitted variables
          and creates column signalizing if a variable has been preselected.
        ----------------------------------------------------
        df: transformed dataset
        key_clmns: list with key columns names
        ---------------------------------------------------- 
        '''
        headers_for_auc = [h for h in df.columns if ((h not in key_clmns) & (h[:2]=="D_"))]
        
        def getauc(var, target, partition):     
            y = np.array(target[partition])
            pred = np.array(var[partition])
            pred = pred.astype(np.float64)
            fpr, tpr, thresholds = metrics.roc_curve(y,pred, pos_label=1)
            return metrics.auc(fpr, tpr)
        
        auc_list_all = []
        parts = ["train","selection"]

        for header in headers_for_auc:
            auc_list_var = [header[2:]]
            # We loop through the two sets ('train' and 'selection') for which an AUC score is needed
            for part in parts:
                auc_value = getauc(var=df[header]
                                   ,target=df['TARGET']
                                   ,partition=df['PARTITION']==part)
                auc_list_var.append(auc_value.round(3)) 
                
            auc_list_all.append(auc_list_var)
             
        df_auc = pd.DataFrame(auc_list_all,columns=['variable','AUC train','AUC selection'])
        
        #Filter based on min AUC
        auc_thresh = df_auc.loc[:,'AUC selection'] > self.preselect_auc
        # We identify those variables for which the AUC score difference between 'train' and 'selection' is within the user-defined ratio
        auc_overtrain = (df_auc.loc[:,'AUC train']*100 - df_auc.loc[:,'AUC selection']*100) < self.preselect_overtrain
        
        # List of variables which passed the two criteria
        df_auc['preselection'] = auc_thresh & auc_overtrain
        
        return df_auc
    
    def _calcCorr(self, df, key_clmns):
        '''
        Method calculates correlation on train set amongst the "D_" variables
        Returns DF with correlations
        ----------------------------------------------------
        df: transformed dataset
        key_clmns: list with key columns names
        ---------------------------------------------------- 
        '''
        headers_for_corr = [h for h in df.columns if ((h not in key_clmns) & (h[:2]=="D_"))]
        
        train = df['PARTITION']=="train"
        dataforcorr = np.transpose(np.matrix(df.loc[train,headers_for_corr],dtype=float))
        with np.errstate(invalid='ignore', divide='ignore'):
            mat_corr = np.corrcoef(dataforcorr)
        
        #Convert numpy to pandas
        df_corr = pd.DataFrame(mat_corr)
        df_corr.columns = headers_for_corr
        df_corr.index = headers_for_corr
        df_corr.fillna(0, inplace=True)
        
        return df_corr
    
                
        
        
        
        

        
        
        
        
        
        
        
        
        