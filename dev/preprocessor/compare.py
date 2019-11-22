#%%
import pandas as pd
import numpy as np
from random import shuffle
from scipy import stats
from typing import Dict, Tuple
import sys

sys.path.insert(0,"C:/Local/pers/Documents/GitHub/Cobra/dev")

import preprocessor.categorical_regrouper as pr

import logging
log = logging.getLogger(__name__)

ROOT = "C:/Local/pers/Documents/GitHub/Cobra/"
df_data = pd.read_csv(ROOT + "datasets/titanic_data.csv")
df_data.rename(columns={'Survived': 'TARGET'}, inplace=True)
df_data['Pclass'] = df_data['Pclass'].astype(object)

split = ['TRAIN']*int(df_data.shape[0]*0.7) + \
        ['TEST']*int(df_data.shape[0]*0.2)+ \
        ['VALIDATION']*int(np.ceil(df_data.shape[0]*0.1))

shuffle(split)

df_data['PARTITION'] = split

df_x = pd.DataFrame(df_data[['Pclass', 'Embarked']][df_data['PARTITION'] == "TRAIN"])
df_y = df_data['TARGET'][df_data['PARTITION'] == "TRAIN"]

#%%
""" NEW SOLUTION """
CR = pr.CategoryRegrouper()

CR.fit(X=df_x, y=df_y, columns=["Embarked", "Pclass"])
print(CR.all_category_map_)
df_new = CR.transform(X=df_x, columns=["Embarked", "Pclass"])

#%%
""" OLD SOLUTION """
def __regroup(var,target,train,pval_thresh=0.01,dummy=True,keep='Missing',rename='Other'):
    '''
    Method regroups categorical variables
    Returns DF mask
    ----------------------------------------------------
    var: input pd.Serie with cat column
    target: pd.Serie with target variable
    train: pd.Serie with parition variable
    pval_thresh: threshold for regrouping
    dummy: scale of booleans (?)
    keep: keep specific groups (?)
    rename: rename the insignificant category
    ---------------------------------------------------- 
    - Each group is tested with a chi² for relevant incidence differences in comparison to a rest-group
    - The rest group has the size of the remaining groups and an 'overall average incidence' (if dummy=True) or 
    - remaining groups average incidence' (if dummy=False)
    - Groups with a pvalue above the threshold are relabled to a single group
    '''
    
    # Define the chi² test condition
    # Groups that do not meet the condition are not analyzed and will be unconditionally relabled
    def _chi2cond_(var=var,target=target,train=train):
        varcounts = var[train].groupby(by=var).count()
        train_inc = target[train].sum()/len(target[train])
        factor = max(train_inc, 1-train_inc)
        analyze_mask = (varcounts*factor)>5
        analyze_groups = analyze_mask.index[analyze_mask].values
        return analyze_groups
    
    # Compute overal incidence mean
    incidence_mean = target[train].mean()
    # Create container of which groups will be kept, compared to the groups which will be relabled
    keepgroups = []
    # Cycle and test each group that meets the chi² condition
    for group in _chi2cond_():
        # Container for target 0/1 observations of the group under scrutiny
        obs_group = []
        # Counts of the target 0/1 occurences for the group under scrutiny
        obs_group.append(((target[train]==0)&(var[train]==group)).sum())
        obs_group.append(((target[train]==1)&(var[train]==group)).sum())
        obs_group = np.array(obs_group)
        # Container for target 0/1 observations of the remaining groups together
        obs_other = []
        # Counts of the target 0/1 occurences for the remaining groups together
        obs_other.append(((target[train]==0)&(var[train]!=group)).sum())
        obs_other.append(((target[train]==1)&(var[train]!=group)).sum())
        obs_other = np.array(obs_other)
        # If dummy=True, we scale the two groups of target 0/1 occurences such that the incidence is equal to the overall incidence
        # The size of the two groups of target 0/1 occurences is still equal to the size of the remaining groups
        if dummy:
            obs_other_size = obs_other.sum()
            obs_other[0]=(1-incidence_mean)*obs_other_size # 0(1) index coincides with target = 0(1)
            obs_other[1]=(  incidence_mean)*obs_other_size
        obs = np.array([obs_group,obs_other])
        # Place at least 1 observation to avoid error in chi2 test
        obs[obs==0] = 1
        # Perform chi² test
        pval = stats.chi2_contingency(obs, correction=False)[1]
        # If pval outperforms threshold, append the group in the keepgroups list
        if pval<=pval_thresh:
            keepgroups.append(group)
        #elif group==keep:
        #    keepgroups.append(group)
    # If the specific group to be kept (e.g. 'Missing') didn't pass the test, append it to the keepgroups list
    if keep not in keepgroups:
        keepgroups.append(keep)
    # Makes a list of all groups not in the keepgroups list
    regroup_mask = [val not in keepgroups for val in var.values]
    var_regroup = var.copy()
    # Rename those groups
    var_regroup[regroup_mask] = rename
    var_regroup.name = "B_"+var.name
    info = (var.name+": from "+str(len(var.unique()))+" to "+str(len(var_regroup.unique())))
    return var_regroup, info

#%%
result = __regroup(var=df_data['Pclass'], #Cabin, Pclass, SibSp, Parch, Embarked
                    target=df_data.loc[:,'TARGET'],
                    train=df_data['PARTITION']=='TRAIN',
                    pval_thresh=0.05,
                    dummy=True,
                    keep='Missing',
                    rename='non-significant')

print(result[0].unique())
print(result[0].head(n=5))
#print(result[1])
df_orig = result[0].to_frame()
df_orig.columns = ["old"]
df_orig["old"] = df_orig["old"].astype('str')
df_orig["old"] = df_orig["old"].astype('category')

df_orig['split'] = df_data['PARTITION']
df_orig = df_orig[df_orig['split'] == 'TRAIN']


#%%
""" COMPARE """
#df_orig.loc[:,"new"] = df_new['Embarked_regrouped'].copy()
df_orig.loc[:,"new"] = df_new['Pclass_regrouped'].copy()

print(df_orig)

df_orig['compare'] = df_orig["new"] == df_orig["old"]

print(df_orig[df_orig['compare'] == False])



#%%


#%%
