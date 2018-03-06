''' 
======================================================================================================================
--------------------------------------------------  MODEL SELECTION  --------------------------------------------
======================================================================================================================
'''
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

class ModelSelection(object):
    '''
    Class for Model Selection
    Finds best model using forward selection
    ----------------------------------------------------
    Author: Jan Benisek, Python Predictions
    Date: 19/02/2018
    ----------------------------------------------------
    ***PARAMETERS***
    :modeling_nsteps:             Size of training set as int <0;1>
    :forced_vars:                 Force variables to be used in forward selection
    :excluded_vars:               List with variables to be excluded
    
    ***ATTRIBUTES***
    :_partition_dict:             Dict with partitioned DFs X/Y train/selection/validation
    :_optimal_nvars:              Optimal number of variables
    ---------------------------------------------------- 
    '''
    
    def __init__(self):
        pass
        
    def fit(self, df_trans, df_unisel, modeling_nsteps, forced_vars, excluded_vars, name):
        '''
        Method fits (=performs) Model Selection
        Returns DF with model performance and list
        ----------------------------------------------------
        df_trans: transformed dataset
        df_unisel: dataframe with univariate selection
        modeling_nsteps: how many variables will be used for modelling
        forced_vars: variables forced to be used in the modelling, list
        excluded_vars: variables to be excluded
        ---------------------------------------------------- 
        '''
        self.modeling_nsteps = modeling_nsteps
        
        ##Create partition
        self._partition_dict = self._getTrainSelectValidXY(df_trans)
        
        ##Perform forward selection
        df_fsel = self._forwardSelection(df_unisel, forced_vars, excluded_vars)
        
        ##Cumulative respone/gain and adds it into df_fsel
        self._cumulatives(df_fsel)
        
        ##Calclates importance and adds it into df_fsel
        self._calcImportance(df_fsel)
        
        ##Give name
        df_fsel.name = name
        
        return df_fsel
        
    def _getTrainSelectValidXY(self, df):
        '''
        Method split given DF into train/test/validation set in respect to X and Y.
        Returns dictionary with DFs
        ----------------------------------------------------
        df: transformed dataset
        ---------------------------------------------------- 
        '''
        
        dvars = [n for n in df.columns if n[:2] == 'D_']
        
        mask_train = df['PARTITION']=="train"
        mask_selection = df['PARTITION']=="selection"
        mask_validation = df['PARTITION']=="validation"
        
        y_train = df.loc[mask_train,'TARGET']
        y_selection = df.loc[mask_selection,'TARGET']
        y_validation = df.loc[mask_validation,'TARGET']
        
        x_train = df.loc[mask_train,dvars]
        x_selection = df.loc[mask_selection,dvars]
        x_validation = df.loc[mask_validation,dvars]
        
        dict_out = {'y_train':y_train, 'y_selection':y_selection, 'y_validation':y_validation, 
                    'x_train':x_train, 'x_selection':x_selection, 'x_validation':x_validation}
        
        return dict_out
    
    def _forwardSelection(self, df_sel, forced_vars, excluded_vars, positive_only=True):
        '''
        Method performs forward selection
        Returns DF with performance
        ----------------------------------------------------
        df_sel: DF with selection from Univariate Selection
        forced_vars: list with varibels forced to be in the forward selection
        excluded_vars: list with variables to be excluded
        positive_only: whether or not all coefs in logit should be positive
        ---------------------------------------------------- 
        '''    
        if not excluded_vars:
            excluded_vars = []
            
        if not forced_vars:
            forced_vars = []
            
        #Sort
        df_sel = df_sel.sort_values(by='AUC selection', ascending=False)
        
        #Build list of variables to be used for Forward selection
        preselected_vars = df_sel['variable'][df_sel['preselection'] == True].tolist()
        preselected_vars = [var for var in preselected_vars if var not in forced_vars+excluded_vars]
        all_vars = ['D_' + var for var in forced_vars + preselected_vars]
        
        df_forward_selection = pd.DataFrame(None,columns=[
                                                          'step',
                                                          'coef',
                                                          'all_coefs_positive',
                                                          'auc_train',
                                                          'auc_selection',
                                                          'auc_validation',
                                                          'predictors_subset',
                                                          'last_var_added',
                                                          'auc_train_rank',
                                                          'selected_model',
                                                          'pred_training',
                                                          'pred_selection',
                                                          'pred_validation'
                                                          ])
        
        
        f_position_forced = lambda i, forced, all_vars: len(forced) if i <= len(forced) else len(all_vars)
        
        n_steps = min(30,len(all_vars))
        predictors = []
        row = 0
        
        #-----------------------------------------------------------------------------------
        #-------------------------------  ITERATE FOR EVERY STEP  --------------------------
        #-----------------------------------------------------------------------------------
        for step in range(1,n_steps):
            
            pos = f_position_forced(step, forced_vars, all_vars)
            remaining_predictors = [var for var in all_vars[:pos] if var not in predictors]
            
            #-----------------------------------------------------------------------------------
            #--------------------------------  FOR EVERY COMBINATION  --------------------------
            #-----------------------------------------------------------------------------------
            for predictor in remaining_predictors:
                predictors_subset = predictors + [predictor]
                #Train - train model
                logit = LogisticRegression(fit_intercept=True, C=1e9, solver = 'liblinear')
                logit.fit(y=self._partition_dict['y_train'], X=self._partition_dict['x_train'][predictors_subset])
                
                #Train - predict and AUC
                y_pred_train = logit.predict_proba(self._partition_dict['x_train'][predictors_subset])
                AUC_train = metrics.roc_auc_score(y_true=self._partition_dict['y_train'], y_score=y_pred_train[:,1])
        
                #Selection - predict and AUC
                y_pred_selection = logit.predict_proba(self._partition_dict['x_selection'][predictors_subset])
                AUC_selection = metrics.roc_auc_score(y_true=self._partition_dict['y_selection'], y_score=y_pred_selection[:,1])
                
                #Validation - predict and AUC
                y_pred_validation = logit.predict_proba(self._partition_dict['x_validation'][predictors_subset])
                AUC_validation = metrics.roc_auc_score(y_true=self._partition_dict['y_validation'], y_score=y_pred_validation[:,1])
                    
                #check if coefs are positive
                all_coefs_positive = (logit.coef_[0] >= 0).all()
                
                #Update DF
                df_forward_selection.loc[row] = [
                                                 step,
                                                 logit.coef_,
                                                 all_coefs_positive,
                                                 AUC_train,
                                                 AUC_selection,
                                                 AUC_validation,
                                                 predictors_subset,
                                                 predictors_subset[-1],
                                                 0,
                                                 False,
                                                 y_pred_train,
                                                 y_pred_selection,
                                                 y_pred_validation
                                                 ]
                row +=1
                
            #Only positive coefs
            if positive_only:
                if len(df_forward_selection[(df_forward_selection['all_coefs_positive'] == True) & (df_forward_selection['step'] == step)]) == 0:
                    raise ValueError("No models with only positive coefficients","NormalStop")
                
                ##Find best model
                #Sort AUC by size
                df_forward_selection['auc_train_rank'] = df_forward_selection.groupby('step')['auc_train'].rank(ascending=False)
                
                #Find model where AUC is highest AND all coefs are positive - convert to boolean flag
                df_forward_selection['selected_model'] = df_forward_selection[df_forward_selection['all_coefs_positive'] == True].groupby(['step'])['auc_train'].transform(max)
                df_forward_selection['selected_model'] = (df_forward_selection['selected_model'] == df_forward_selection['auc_train'])
            else:
                ##Highest AUC, regardless of coefs
                df_forward_selection['selected_model'] = (df_forward_selection.groupby(['step'])['auc_train'].transform(max) == df_forward_selection['auc_train'])
                
            ##Add next predictor
            add_variable = df_forward_selection.loc[(df_forward_selection['selected_model'] == True) & (df_forward_selection['step'] == step), 'last_var_added'].iloc[0]
            predictors.append(add_variable)
            
        #Return only DF with selected models
        clmns_out = ['step', 'coef', 'auc_train', 'auc_selection', 'auc_validation', 'predictors_subset', 'last_var_added',
                     'pred_training','pred_selection','pred_validation']
        
        df_out = df_forward_selection[clmns_out][df_forward_selection['selected_model'] == True]

        
        #Reset index - otherwise lots of nasty errors later
        df_out.reset_index(inplace=True, drop=True)
        
        return df_out
        
    def _cumulatives(self, df):
        '''
        Method calculates cumulative gains/response
        Returns nothing, adds cgains/response into the dataframe
        ----------------------------------------------------
        df: df with best models 
        ---------------------------------------------------- 
        '''    
        
        def cumulatives(y,yhat,perc_as_int=False,dec=2):
            nrows = len(y)
            npositives = y.sum()
            y_yhat = pd.DataFrame({"y":y, "yhat":yhat}).sort_values(by='yhat', ascending=False).reset_index(drop=True)
            cresp = []
            cgains = [0]
            for stop in (np.linspace(0.01,1,100)*nrows).astype(int):
                cresp.append(round(y_yhat.loc[:stop,'y'].mean()*max(100*int(perc_as_int),1),dec))
                cgains.append(round(y_yhat.loc[:stop,'y'].sum()/npositives*max(100*int(perc_as_int),1),dec))
            return cresp,cgains
        
        cresp_all = []
        cgains_all = []
        
        for i in range(0,len(df)):

            out = cumulatives(y=self._partition_dict['y_selection'],
                              yhat=df.iloc[i]['pred_selection'][:,1],
                              perc_as_int=True,
                              dec=2)
            cresp_all.append(out[0]) 
            cgains_all.append(out[1])
         
        #Add it to the models dataframe
        df['cum_response'] = cresp_all
        df['cum_gains'] = cgains_all
        
    def _calcImportance(self, df):
        '''
        Method calculates importance of each variable
        Returns nothing, adds importnace into the dataframe
        ----------------------------------------------------
        df: df with best models 
        ---------------------------------------------------- 
        ''' 
        importance_all = []
        for row in df.index:
            importance_dict = {}
            for pr in df.iloc[row,:]['predictors_subset']:
                corr = stats.pearsonr(self._partition_dict['x_selection'].loc[:,pr].values, df.iloc[row,:]['pred_selection'][:,1])
                importance_dict[pr[2:]] = corr[0]
            
            importance_all.append(importance_dict) 
        
        #Add it to the models dataframe
        df['importance'] = importance_all 
