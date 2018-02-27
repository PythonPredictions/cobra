''' 
======================================================================================================================
---------------------------------------------------------  TESTING  ---------------------------------------------------
======================================================================================================================
'''
import sys
sys.path.append('C:/Local/pers/Documents/GitHub/COBRA/source_code')

import pandas as pd

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
                            forced_vars=None,
                            excluded_vars=None,
                            name='All variables')

build.plotAUC(df_model1)
build.plotVariableImportance(df_model1)
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
    
    

    
    
    

    












































    
    








