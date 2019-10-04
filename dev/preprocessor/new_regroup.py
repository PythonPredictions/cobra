#%%
import pandas as pd
import numpy as np
from random import shuffle
from scipy import stats

ROOT = "C:/Local/pers/Documents/GitHub/Cobra/"
df_data = pd.read_csv(ROOT + "datasets/titanic_data.csv")
df_data.rename(columns={'Survived': 'TARGET'}, inplace=True)

split = ['TRAIN']*int(df_data.shape[0]*0.5) + \
        ['TEST']*int(df_data.shape[0]*0.2)+ \
        ['VALIDATION']*int(np.ceil(df_data.shape[0]*0.3))

shuffle(split)

df_data['PARTITION'] = split

#%%
''' ORIGINAL CODE '''
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
''' RUN ORIGINAL CODE '''
result = __regroup(var=df_data['Embarked'], #Cabin, Pclass, SibSp, Parch, Embarked
                    target=df_data.loc[:,'TARGET'],
                    train=df_data['PARTITION']=='TRAIN',
                    pval_thresh=0.05,
                    dummy=True,
                    keep='Missing',
                    rename='Non-significants')

print(result[0].unique())
print(result[0].head(n=5))
print(result[1])
df_tst = result[0]

#%%
''' TEST CHISQR CONDITION '''
def _chi2cond_(var,target,train):
    #simple group by - pandas series
    varcounts = var[train].groupby(by=var).count()
    #train incidence - 0.3775280898876405
    train_inc = target[train].sum()/len(target[train])
    #Why? -0.6224719101123595
    factor = max(train_inc, 1-train_inc)
    #which groups to analyze - boolean
    analyze_mask = (varcounts*factor)>5
    #filter groups to be kept - array([0, 1, 2], dtype=int64)
    analyze_groups = analyze_mask.index[analyze_mask].values
    return analyze_groups

chi = _chi2cond_(var=df_data['Embarked'],
                target=df_data.loc[:,'TARGET'],
                train=df_data['PARTITION']=='TRAIN')

#%%
varcounts = df_data['Parch'][df_data['PARTITION']=='TRAIN'].groupby(by=df_data['Parch']).count()
train_inc = df_data.loc[:,'TARGET'][df_data['PARTITION']=='TRAIN'].sum()/len(df_data.loc[:,'TARGET'][df_data['PARTITION']=='TRAIN'])
factor = max(train_inc, 1-train_inc)
analyze_mask = (varcounts*factor)>5
analyze_groups = analyze_mask.index[analyze_mask].values

#%%
df_data['TARGET'][df_data['PARTITION']=='TRAIN'].mean()

#%%
''' TEST TESTING '''
target = df_data.loc[:,'TARGET']
train = df_data['PARTITION']=='TRAIN'
var = df_data['Embarked']

for group in chi:
    group == 'S'
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

#S
#obs_group
#array([225,  95], dtype=int64)
#
#obs_other
#array([58, 67], dtype=int64)

#%%
pd.crosstab(df.regiment, df_data.loc[:,'TARGET'], margins=True)

#%%
incidence_mean = target[train].mean()
dummy=True

if dummy:
    obs_other_size = obs_other.sum() #400
    obs_other[0]=(1-incidence_mean)*obs_other_size # 0(1) index coincides with target = 0(1)
    obs_other[1]=(  incidence_mean)*obs_other_size
obs = np.array([obs_group,obs_other])
# Place at least 1 observation to avoid error in chi2 test
obs[obs==0] = 1
# Perform chi² test
pval = stats.chi2_contingency(obs, correction=False)[1]

#obs
#array([[ 19,  26],
#       [248, 151]], dtype=int64)

#%%
