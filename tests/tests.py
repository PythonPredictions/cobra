''' 
======================================================================================================================
---------------------------------------------------------  TESTING  ---------------------------------------------------
======================================================================================================================
'''
import sys
sys.path.append('C:/Local/pers/Documents/GitHub/COBRA/source_code')

import pandas as pd
pd.set_option("display.max_columns",50)

data_path = 'C:/Local/pers/Documents/GitHub/COBRA/datasets/data.csv'
data_types_path = 'C:/Local/pers/Documents/GitHub/COBRA/datasets/data_types.csv'

'''
TO-DO

'''

'''===================  DATA PREPARATION  ==================='''
import classes.data_preparation as dpc

dprep = dpc.DataPreparation(data_path,
                            data_types_path,
                            partition_train=0.6,
                            partition_select=0.3,
                            partition_valid=0.1,
                            sampling_1=1,
                            sampling_0=1,
                            discret_nbins=5,
                            regroup_sign=0.001)

df_prep = dprep.transform()  

#Attributes
dict_headers = dprep._headers_dict
dict_part = dprep._partitioning_settings
dict_sample = dprep._sampling_settings

'''===================  UNIVARIATE SELECTION  ==================='''
import classes.univariate_selection as us

unisel = us.UnivariateSelection(preselect_auc=0.53, 
                                preselect_overtrain=5)
df_sel, df_corr = unisel.fit(df_prep)

'''===================  MODEL SELECTION  ==================='''
import classes.model_selection as ms
modsel = ms.ModelSelection()

df_models1 = modsel.fit(df_prep, 
                        df_sel,
                        modeling_nsteps=30,
                        forced_vars=None,
                        excluded_vars=None,
                        name='IamOriginal')

df_models2 = modsel.fit(df_prep, 
                        df_sel,
                        modeling_nsteps=30,
                        forced_vars=['scont_3'],
                        excluded_vars=None,
                        name='IamNew')

partition = modsel._partition_dict


'''===================  TEST COBRA INTERFACE ==================='''
import cobra.cobra as c

build = c.COBRA(data_path,
                data_types_path,
                partition_train=0.6,
                partition_select=0.3,
                partition_valid=0.1,
                sampling_1=1,
                sampling_0=1,
                discret_nbins=5,
                regroup_sign=0.001)
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
                            forced_vars=['scat_3'],
                            excluded_vars=None,
                            name='All variables')

build.plotAUC(df_model1)
build.plotVariableImportance(df_model1, 4)
build.plotCumulatives([df_model1], df_transformed)

#second model
df_model2 = build.fit_model(df_transformed, 
                            df_unisel,
                            modeling_nsteps=30,
                            forced_vars=None,
                            excluded_vars=['age','relationship', 'sex'],
                            name='Experiment')

build.plotAUC(df_model2)
build.plotVariableImportance(df_model2)
build.plotCumulatives([df_model2], df_transformed)

#Model comparison
build.plotAUCComparison([df_model1, df_model2])
build.plotCumulatives([df_model1, df_model2], df_transformed)
    



from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np

selected_variables = ['D_scat_3', 'D_relationship', 'D_marital-status', 'D_occupation', 
                      'D_education', 'D_education-num', 'D_age', 'D_hours-per-week', 
                      'D_sex', 'D_capital-gain', 'D_workclass', 'D_race']


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


def __buildModel(predictors_subset):
    '''
    Method buils Logistic Regression with the given subset of variables.
    Returns trained model and fit
    ----------------------------------------------------
    predictors_subset: list of variables to be used in the model
    ---------------------------------------------------- 
    '''
    
    # Fit model on predictors_subset and retrieve performance metric
    model = LogisticRegression(fit_intercept=True, C=1e9, solver = 'liblinear')
    modelfit = model.fit(y=_partition_dict['y_train'], X=_partition_dict['x_train'][predictors_subset])
    
    # Position of the TARGET==1 class
    pos = [i for i,h in enumerate(modelfit.classes_) if h==1]
    # Prediction probabilities for the TARGET==1
    y_pred = modelfit.predict_proba(_partition_dict['x_train'][predictors_subset])[:,pos]
    auc = metrics.roc_auc_score(y_true=_partition_dict['y_train'], y_score=y_pred)
    
    model_score_dict = {"modelfit":modelfit,
                        "auc":auc,
                        "predictor_names":predictors_subset,
                        "predictor_lastadd":predictors_subset[-1]}
    
    return model_score_dict




for x,y,part in [(_partition_dict['x_train'], _partition_dict['y_train'], 'train'),
                 (_partition_dict['x_selection'], _partition_dict['y_selection'], 'selection'),
                (_partition_dict['x_validation'], _partition_dict['y_validation'], 'validation')]:
    print(x)
    


def __calculateAUC(df):
    '''
    Method for computing AUC of all sets (train, selection & validation)
    Returns DF with the AUC
    ----------------------------------------------------
    df: df without AUC
    partition: dictionary with DFs of partitions (train/sel/valid X/Y)
    ---------------------------------------------------- 
    '''
    df = df[:]
    for x,y,part in [(_partition_dict['x_train'], _partition_dict['y_train'], 'train'),
                     (_partition_dict['x_selection'], _partition_dict['y_selection'], 'selection'),
                     (_partition_dict['x_validation'], _partition_dict['y_validation'], 'validation')]:
        pos = [i for i,h in enumerate(df.modelfit.classes_) if h==1]
        y_pred = df.modelfit.predict_proba(x[df['predictor_names']])[:,pos]
        df["auc_"+part] = metrics.roc_auc_score(y_true=y, y_score=y_pred)
        df["pred_"+part] = y_pred
        
    return df

def __forward(current_predictors, pool_predictors, positive_only=True):
    '''
    Method for forward selection
    Returns best model
    ----------------------------------------------------
    current_predictors: current predictors to be used = predictors
    pool_predictors: remaining predictors = selected_variables
    positive_only: predictors must be positivee (???)
    ---------------------------------------------------- 
    '''
    #Pull out predictors we still need to process
    remaining_predictors = [p for p in pool_predictors if p not in current_predictors]
    # If there are no more predictors left to use, raise an error we can easily identify as normal
    if len(remaining_predictors)==0:
        raise ValueError("No more predictors left to use","NormalStop")
    
    #Create a model for each combination of: current predictor(s) + one of the remaining predictors
    #Keep track of the submodels and their performance
    #If error skip to next and do not include in comparison table
    results = []
    errorcount = 0
    for p in remaining_predictors:
        try:
            results.append(__buildModel(current_predictors+[p]))
        except:
            errorcount += 1 
    df_results = pd.DataFrame(results)
    
    # If we require all coefficients to be positive...
    if positive_only:
        #Create a flag for each submodel to test if all coefficients are positive 
        all_positive = pd.Series(None, index=df_results.index)
        for i in range(0,len(df_results)):
            all_positive[i] = (df_results.modelfit[i].coef_ >= 0 ).all()
            
        # if no model exist with only positive coefficients raise error we can easily identify as normal
        if (all_positive==0).all():
            raise ValueError("No models with only positive coefficients","NormalStop")
            
        #Choose model with best performance and only positive coefficients
        best_model = df_results.loc[df_results[all_positive==1].auc.idxmax()]
        df_best_model = __calculateAUC(best_model)
        
    # If we don't require all coefficients to be positive...   
    else:
        #Choose model with best performance
        best_model = df_results.loc[df_results.auc.idxmax()]
        df_best_model = __calculateAUC(best_model)
    
    return df_best_model



df_sel = df_unisel
forced_vars = ['scat_3', 'scat_4']
excluded_vars = None


df_best_models = pd.DataFrame(columns=["modelfit",
                                       "predictor_names",
                                       "predictor_lastadd",
                                       "auc_train",
                                       "auc_selection",
                                       "auc_validation",
                                       "pred_train",
                                       "pred_selection",
                                       "pred_validation"])
df_sel_aux = df_sel.copy()
#Set variables to be used for the selection
if forced_vars:
    #Flag those which user chose to be in the model
    df_sel_aux['forced'] = np.where(df_sel_aux['variable'].isin(forced_vars), True, False)
    #Sort the variables so the one which user chose are first, then the ones from preselection
    # and those are sorted by train AUC
    df_sel_aux.sort_values(['forced','preselection', 'AUC train'], ascending=[False, False, False], inplace=True)
    #Flag for which variables will be used
    df_sel_aux['final_vars'] = np.where(((df_sel_aux['preselection'] == True) | (df_sel_aux['forced'] == True)),
                                          True,
                                          False)
    selected_variables = df_sel_aux['variable'][df_sel_aux['final_vars'] == True].tolist()
else:
    df_sel_aux['forced'] = False
    df_sel_aux.sort_values(['preselection', 'AUC train'], ascending=[False, False], inplace=True)
    selected_variables = df_sel_aux['variable'][df_sel_aux['preselection'] == True].tolist()
    
if excluded_vars:
    selected_variables = [var for var in selected_variables if var not in excluded_vars]


selected_variables = ['D_' + var for var in selected_variables]

n_steps = min(30,len(selected_variables))
predictors = []
f_position_forced = lambda i, forced, all_vars: len(forced) if i <= len(forced) else len(all_vars)

for i in range(1,n_steps+1):
    pos = f_position_forced(i, forced_vars, selected_variables)
    print('******************Iter {} ********************************'.format(i))
    print('---------USE PREDICTORS:---------')
    print(selected_variables[:pos])
    print('---------PREDICTORS:---------')
    print(predictors)
    result = __forward(current_predictors=predictors,
                       pool_predictors=selected_variables[:pos],
                       positive_only=True)
    print(result)
    df_best_models.loc[i] = result
    predictors = df_best_models.loc[i].predictor_names

        

#df_best_models.reset_index(inplace=True, drop=True)
    
    
''' 
=============================================================================================================
=============================================================================================================
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
    excluded_vars = []

#Sort
df_sel = df_sel.sort_values(by='AUC selection', ascending=False)

#Build list of variables to be used for Forward selection
preselected_vars = df_sel['variable'][df_sel['preselection'] == True].tolist()
preselected_vars = [var for var in preselected_vars if var not in forced_vars+excluded_vars]
all_vars = ['D_' + var for var in forced_vars + preselected_vars]



''' 
------------------  MAIN LOOP  ------------------
'''
df_forward_selection = pd.DataFrame(None,columns=['step',
                                             'coef',
                                             'AUC_train',
                                             'predictors_subset',
                                             'last_var_added'])
        
f_position_forced = lambda i, forced, all_vars: len(forced) if i <= len(forced) else len(all_vars)

n_steps = min(30,len(all_vars))
predictors = []
row = 0

for step in range(1,n_steps+1):
    print('*******************Iter {}*******************'.format(step))
    
    pos = f_position_forced(step, forced_vars, all_vars)
    remaining_predictors = [var for var in all_vars[:pos] if var not in predictors]
    
    for predictor in remaining_predictors:
        predictors_subset = predictors + [predictor]
        #Train model
        logit = LogisticRegression(fit_intercept=True, C=1e9, solver = 'liblinear')
        logit.fit(y=_partition_dict['y_train'], X=_partition_dict['x_train'][predictors_subset])
        #Predict
        y_pred_train = logit.predict_proba(_partition_dict['x_train'][predictors_subset])
        AUC_train = metrics.roc_auc_score(y_true=_partition_dict['y_train'], y_score=y_pred_train[:,1])
        
        
        df_forward_selection.loc[row] = [step, logit.coef_, AUC_train, predictors_subset, predictors_subset[-1]]
        
        row +=1
     
        
        
   


    

    
    
    

    







































    
    









