''' 
======================================================================================================================
---------------------------------------------------------  TESTING  ---------------------------------------------------
======================================================================================================================
This is my (Honza) script to test and develop in Cobra
import sys
sys.path.append('C:/Local/pers/Documents/GitHub/COBRA/source_code')
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option("display.max_columns",50)

data_path = 'C:/Local/pers/Documents/GitHub/COBRA/datasets/data.csv'
data_types_path = 'C:/Local/pers/Documents/GitHub/COBRA/datasets/data_types.csv'

'''
TO-DO
-sometimes error - no variables with positive coef. Even if error is thrown, return stuff!
-the functions can be applied in a vectorized way
-further improve the forward selection

TEST
'''


'''===================  TEST COBRA INTERFACE ==================='''
import cobra.cobra as c

build = c.COBRA(data_path,
                data_types_path,
                partition_train=0.5,
                partition_select=0.3,
                partition_valid=0.2,
                sampling_1=1,
                sampling_0=1,
                discret_nbins=5,
                regroup_sign=0.001,
                rseed=0)
df_transformed = build.transform()


#I want to try more unisel
df_unisel, df_corr = build.fit_univariate(df_transformed,
                                          preselect_auc=0.53, 
                                          preselect_overtrain=5)

build.plotPredictorQuality(df_unisel)
build.plotCorrMatrix(df_corr)
build.plotIncidence(df_transformed, 'age')

#I want to try more models
#first model
df_model1 = build.fit_model(df_transformed, 
                            df_unisel,
                            modeling_nsteps=30,
                            forced_vars=['scont_1', 'scont_2'],
                            excluded_vars=None,
                            name='All variables')

build.plotAUC(df_model1)
build.plotVariableImportance(df_model1, 5)
build.plotCumulatives([(df_model1,3)], df_transformed)


#second model
df_model2 = build.fit_model(df_transformed, 
                            df_unisel,
                            modeling_nsteps=30,
                            forced_vars=None,
                            excluded_vars=None,
                            name='Experiment')

build.plotAUC(df_model2)
build.plotVariableImportance(df_model2, 6)
build.plotCumulatives([(df_model2, 5)], df_transformed)

#Model comparison
build.plotAUCComparison([(df_model1,3), (df_model2,5)])
build.plotCumulatives([(df_model1,3), (df_model2,5)], df_transformed)

'''===================  PROFILING & SPEED ==================='''
import cProfile
import timeit

cProfile.run('build.fit_univariate(df_transformed,preselect_auc=0.53, preselect_overtrain=5)')

def classWrapper():
    build = c.COBRA(data_path,
                data_types_path,
                partition_train=0.5,
                partition_select=0.3,
                partition_valid=0.2,
                sampling_1=1,
                sampling_0=1,
                discret_nbins=5,
                regroup_sign=0.001,
                rseed=0)
    df_transformed = build.transform()
    
    df_unisel, df_corr = build.fit_univariate(df_transformed,
                                          preselect_auc=0.53, 
                                          preselect_overtrain=5)
    
    df_model1 = build.fit_model(df_transformed, 
                            df_unisel,
                            modeling_nsteps=30,
                            forced_vars=None,
                            excluded_vars=None,
                            name='All variables')

%timeit classWrapper()
'''
Improvements:
-------------
- 06/03/2018, JBE
  11 s ± 267 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


'''

    

def _getTrainSelectValidXY(df):
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

_partition_dict = _getTrainSelectValidXY(df_transformed)

''' 
=============================================================================================================
======================================= FORWARD SELECTION ===================================================
=============================================================================================================
-Only boolean target
'''
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np

df_sel = df_unisel
forced_vars = ['scont_1', 'scont_2']
excluded_vars = None
positive_only = True

#if None, replace by empty list
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



''' 
------------------  MAIN LOOP  ------------------
'''
df_forward_selection = pd.DataFrame(None,columns=[
                                                  'step',
                                                  'coef',
                                                  'all_coefs_positive',
                                                  'AUC_train',
                                                  'AUC_selection',
                                                  'AUC_validation',
                                                  'predictors_subset',
                                                  'last_var_added',
                                                  'AUC_train_rank',
                                                  'selected_model',
                                                  'pred_training',
                                                  'pred_selection',
                                                  'pred_validation'
                                                  ])
        
f_position_forced = lambda i, forced, all_vars: len(forced) if i <= len(forced) else len(all_vars)

n_steps = min(30,len(all_vars))
predictors = []
row = 0

for step in range(1,n_steps):
    print('*******************Iter {}*******************'.format(step))
    
    pos = f_position_forced(step, forced_vars, all_vars)
    remaining_predictors = [var for var in all_vars[:pos] if var not in predictors]
    
    for predictor in remaining_predictors:
        predictors_subset = predictors + [predictor]
        #Train - train model
        logit = LogisticRegression(fit_intercept=True, C=1e9, solver = 'liblinear')
        logit.fit(y=_partition_dict['y_train'], X=_partition_dict['x_train'][predictors_subset])
        
        #Train - predict and AUC
        y_pred_train = logit.predict_proba(_partition_dict['x_train'][predictors_subset])
        AUC_train = metrics.roc_auc_score(y_true=_partition_dict['y_train'], y_score=y_pred_train[:,1])
        
        #Selection - predict and AUC
        y_pred_selection = logit.predict_proba(_partition_dict['x_selection'][predictors_subset])
        AUC_selection = metrics.roc_auc_score(y_true=_partition_dict['y_selection'], y_score=y_pred_selection[:,1])
        
        #Validation - predict and AUC
        y_pred_validation = logit.predict_proba(_partition_dict['x_validation'][predictors_subset])
        AUC_validation = metrics.roc_auc_score(y_true=_partition_dict['y_validation'], y_score=y_pred_validation[:,1])
        
        #check if coefs are positive
        all_coefs_positive = (logit.coef_[0] >= 0).all()
        
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
        df_forward_selection['AUC_train_rank'] = df_forward_selection.groupby('step')['AUC_train'].rank(ascending=False)
        
        #Find model where AUC is highest AND all coefs are positive - convert to boolean flag
        df_forward_selection['selected_model'] = df_forward_selection[df_forward_selection['all_coefs_positive'] == True].groupby(['step'])['AUC_train'].transform(max)
        df_forward_selection['selected_model'] = (df_forward_selection['selected_model'] == df_forward_selection['AUC_train'])
    else:
        ##Highest AUC, regardless of coefs
        df_forward_selection['selected_model'] = (df_forward_selection.groupby(['step'])['AUC_train'].transform(max) == df_forward_selection['AUC_train'])
        
    ##Add next predictor
    add_variable = df_forward_selection.loc[(df_forward_selection['selected_model'] == True) & (df_forward_selection['step'] == step), 'last_var_added'].iloc[0]
    predictors.append(add_variable)
    
    clmns_out = ['step', 'coef', 'AUC_train', 'AUC_selection', 'AUC_validation', 'predictors_subset', 'last_var_added',
                 'pred_training','pred_selection','pred_validation']

df_tst = df_forward_selection[clmns_out][df_forward_selection['selected_model'] == True]




''' 
=============================================================================================================
=========================================== CSV IMPORT ======================================================
=============================================================================================================
-Only boolean target
'''




#Loads Data types
types_exist = True   

#load data_types
try:
    df_types = pd.read_csv(data_types_path, header=None)
    df_types.columns = ['variable','data_type']
except FileNotFoundError:
    types_exist = False
    df_types = pd.DataFrame()
  
#load data
df = pd.read_csv(data_path, header=0, sep=None, engine='python')

   
#change datatypes
if types_exist: 
    for row in df_types.itertuples(): #0:index, 1:variable, 2:data_type        
        if row[2] == 'int':
            df[row[1]] = df[row[1]].astype(np.int64)
        if row[2] in ['str', 'bool']:
            df[row[1]] = df[row[1]].apply(str)

return df, df_types



types_exists = True
data_types_path = 'C:/Local/pers/Documents/GitHub/COBRA/datasets/data_types.csv'

if types_exists:
    try:
        df_dtypes = pd.read_csv(data_types_path, header=None)
        df_dtypes.columns = ['variable','data_type']
        
        
        
        dict_dtypes = {k: getattr(__builtins__, v) for k, v in df_dtypes.to_dict('dict')['data_type'].items()}
    except FileNotFoundError:
        raise 
        
    df = pd.read_csv(data_path, header=0, sep=',', engine='c', converters=dict_dtypes)
    
else:
    
        
  
set(np.unique(df_dtypes['data_type'])) < set(['int','str','bool'])










      
        
df_types = pd.read_csv(data_types_path, header=None)
df_types.columns = ['variable','data_type']
dict_types = df_types.to_dict('dict')

tst = dict_types['data_type']

df = pd.read_csv(data_path, header=0, sep=None, engine='python', converters=dict_types['data_type'])
    
string = 'str'
string = string[1:]
print(string)


import re
string = 'str'
string = re.sub(r'^"|"$', '', string)
print(string)

d2 = {k: re.sub(r'^"|"$', '', v) for k, v in tst.items()}
d2 = {k: getattr(__builtins__, v) for k, v in tst.items()}



















