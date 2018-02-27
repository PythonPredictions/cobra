
# coding: utf-8

# # Univariate analysis

# ### General Imports

# In[1]:

import math
import csv
import warnings
import time
import os
import itertools
import scipy.integrate
import re


# In[2]:

import numpy as np
import pandas as pd


# In[3]:

from scipy import stats
from itertools import chain
from sklearn import metrics


# ---

# ### Miscellaneous

# In[4]:

log = []


# In[5]:

# When code is in script, we define the path of the script's parent folder location as the 'root' directory
# From this 'root' we can travel to the relevant folders with minimal adjustment
try:
    root = os.path.dirname(os.path.realpath(__file__))
    root = "/".join(root.split('\\')[:-1])
    log.append('Dynamic paths'+'\n')
except:
    root = 'C:/wamp64/www/python_predictions_4/assets/scripts'
    log.append('Static paths'+'\n')


# In[6]:

# To allow pandas dataframes to display more columns
pd.set_option("display.max_columns",50)


# ---

# ### Read data and organize

# Basetable and its types

# In[7]:

# A Types csv file CAN be defined to be used to convert variables (of the basetable, see below) to the desired data types
# The Types csv files should include one column with variable names and one column with desired types (e.g. int,float,str,bool)
# If no Types csv file is provided no convertions will be forced. In that case 'Python' will guess the data type of each column 
types_path = root+"/python/data_types.csv"
types_exist = True

try:    
    df_types = pd.read_csv(types_path, header=None)
    bool_mask = df_types[1]!='bool'
    # Extract the functions based on the given type (e.g. 'str' -> str, 'int' -> int), for proper convertion 
    df_types.loc[bool_mask,1] = [getattr(__builtins__, type_str) for type_str in df_types.loc[bool_mask,1]]
    # A type 'bool' is also attributed the function str, for convertion
    df_types.loc[bool_mask==False,1] = getattr(__builtins__, 'str')
    #types = df_types[bool_mask].set_index(0).T.to_dict('records')
    types = df_types.set_index(0).T.to_dict('records')
except FileNotFoundError:
    types = [dict()]
    types_exist = False


# In[8]:

# The basetable csv file should have column names as its first row
# The columns names should include 'TARGET', 'ID'
data_path = root+"/python/data.csv"

df_in = pd.read_csv(data_path
                    ,header=0
                    ,sep=None
                    ,engine='python'
                    ,converters=types[0])

# If no Types csv file was provided pd.read_csv guessed the types, we now output these types in a csv for re-use & later use
if types_exist == False:
    filename = root+"/python/data_types.csv"
    funtotype = lambda x:re.findall('[a-z]+',str(x))[0].replace('object','str')
    with open(filename, 'w') as csvfile:
        write=csv.writer(csvfile, delimiter =',')
        write.writerows([column
                         ,funtotype(df_in[column].dtype)] for column in df_in.columns)


# In[9]:

# Function to remove quotes from variable names and/or variable values
def strip_quot(x_in):
    try:
        x_out = x_in.strip().strip('"').strip("'")
    except:
        x_out=x_in
    return x_out

# Function to put 'id' and 'target' variable names in uppercase, all other variable names are put in lowercase
# This is coded as to visually differentiate predictors from other variables
# But another combination of upper/lower is possible as well, e.g. all variable names in uppercase
def lower_upper(x_in):
    if ((x_in.lower() == 'id')|(x_in.lower() == 'target')):
        x_out = x_in.upper()
    else:
        x_out = x_in.lower()
    return x_out

# Function to group variable names based on the data type of the variable
# Could as well use the types in Types.csv
def get_headers(dataframe,type): 
    return dataframe.select_dtypes(include=[type]).columns.values


# In[10]:

# Clean up quotes from column names
df_in = df_in.rename(columns=strip_quot)

# Perform uppercase/lowercase transformation to column names
df_in = df_in.rename(columns=lower_upper)

# Clean up quotes from column values
df_in = df_in.applymap(strip_quot)


# In[11]:

# Group variable (names) based on the respective data type of each variable
# With this information we know which variables are destined for equifrequency, regrouping or simply passing (see further)
other_headers = [n for n in ["TARGET","ID"]]
try:
    bool_headers = [n for n in df_types.loc[bool_mask==False,0].values if n not in other_headers]
except:
    bool_headers = []
object_headers = [n for n in get_headers(df_in,'object') if n not in other_headers+bool_headers]
numeric_headers = [n for n in get_headers(df_in,'number') if n not in other_headers+bool_headers]


# Analysis settings

# In[12]:

# Import settings defined by the user
df_settings = pd.read_csv(root+'/python/analysis_settings.csv', sep=',', index_col=0, header=None).T


# ---

# ### Partitioning 
# Shuffle and sort on TARGET
df_in = df_in.iloc[np.random.permutation(len(df_in))].sort_values(by='TARGET', ascending=False).reset_index(drop=True)

# Create dic of partitioning settings
partition_dic = {'train':df_settings.loc[:,'partitioning_train']
                ,'selection':df_settings.loc[:,'partitioning_selec']
                ,'validation':df_settings.loc[:,'partitioning_valid']
                }

# Create a partition variable
partition = []
sorted_target=df_in.TARGET #Just the target since it is allready sorted (see above)
for target in [sorted_target.iloc[0],sorted_target.iloc[-1]]:
    target_length = (sorted_target==target).sum()
    for part in partition_dic:
        partition.extend( [part]*math.ceil(target_length*partition_dic[part][1]/100) )

# Attach partition variable to dataframe
df_in["PARTITION"] = partition[:len(df_in)]

# Sampling based on analysis settings (if both sampling_settings are set to 100, all data is used)
sampling_settings = {1:df_settings.sampling_1, 0:df_settings.sampling_0}
drop_index = []
for sample in sampling_settings:
    if sampling_settings[sample].values<100:
        sample_length = int(round((df_in.TARGET==sample).sum() * sampling_settings[sample]/100))
        for part in partition_dic:
            part_length = int(round(sample_length * partition_dic[part] / 100))
            drop_index_part = df_in[(df_in.TARGET==sample) & (df_in.PARTITION==part)].index[part_length:]
            drop_index.extend(drop_index_part)
        #drop_index = df_in[df_in.TARGET==sample].index[sample_length:]
df_in.drop(drop_index,inplace=True)
df_in.reset_index(drop=True, inplace=True)


# ---

# ### Output Container

# In[15]:

# Create output dataframe which will contain transformed variables
df_out = df_in.loc[:,["ID","TARGET","PARTITION"]].copy()


# ---

# ### Preprocessing of continuous variables

# Discretization function for Continuous variables

# In[16]:

### This function is a reworked version of pd.qcut to satisfy our particular needs
### Takes for var a continuous pd.Series as input and returns a pd.Series with bin-labels (e.g. [4,6[ )
### Train takes a series/list of booleans (note: we define bins based on the training set)
### Autobins reduces the number of bins (starting from nbins) as a function of the number of missings
### Nbins is the wished number of bins
### Precision=0 results in integer bin-labels if possible
### twobins=True forces the function to output at least two bins
### catchLarge tests if some groups (or missing group) are very large, and if so catches and outputs two groups
#### note: catchLarge makes twobins irrelevant

def eqfreq(var, train, autobins=True, nbins=10, precision=0, twobins=True, catchLarge=True):
    
    
    # Test for large groups and if one exists pass them with two bins: Large_group,Other
    if catchLarge:
        catchPercentage=1-(1/nbins)
        groupCount = var[train].groupby(by=var[train]).count()
        maxGroupPerc = groupCount.max()/len(var[train])
        missingPerc = sum(var[train].isnull())/len(var[train])
        if maxGroupPerc>=catchPercentage:
            largeGroup = groupCount.sort_values(ascending=False).index[0]
            x_binned = var.copy()
            x_binned.name = 'B_'+var.name
            x_binned[x_binned!=largeGroup]='Other'
            cutpoints=None
            info = (var.name+": One large group, outputting 2 groups")
            return x_binned, cutpoints, info
        elif missingPerc>=catchPercentage:
            x_binned = var.copy()
            x_binned.name = 'B_'+var.name
            x_binned[x_binned.isnull()]='Missing'
            x_binned[x_binned!='Missing']='Other'
            cutpoints=None
            info = (var.name+": One large missing group, outputting 2 groups")
            return x_binned, cutpoints, info
    # Adapt number of bins as a function of number of missings
    if autobins:
        length = len(var[train])
        missing_total = var[train].isnull().sum()
        missing_perten = missing_total/length*10
        nbins = max(round(10-missing_perten)*nbins/10 ,1)
    # Store the name and index of the variable
    name = var.name
    series_index = var.index
    # Transform var and train to a np.array and list respectively, which is needed for some particular function&methods
    x = np.asarray(var)
    train = list(train)
    # First step in finding the bins is determining what the quantiles are (named as cutpoints)
    # If the quantile lies between 2 points we use lin interpolation to determine it
    cutpoints = var[train].quantile(np.linspace(0,1,nbins+1),interpolation = 'linear')
    # If the variable results only in 2 unique quantiles (due to skewness) increase number of quantiles until more than 2 bins can be formed
    if twobins:
        extrasteps = 1
        # Include a max. extrasteps to avoid infinite loop
        while (len(cutpoints.unique())<=2) & (extrasteps<20):
            cutpoints = var[train].quantile(np.linspace(0,1,nbins+1+extrasteps),interpolation = 'linear')
            extrasteps+=1
    # We store which rows of the variable x lies under/above the lowest/highest cutpoint 
    # Without np.errstate(): x<cutpoints.min() or x>cutpoints.max() can give <RuntimeWarning> if x contains nan values (missings)
    # However the function will result in False in both >&< cases, which is a correct result, so the warning can be ignored
    with np.errstate(invalid='ignore'):
        under_lowestbin = x < cutpoints.min()
        above_highestbin= x > cutpoints.max()


    def _binnedx_from_cutpoints(x, cutpoints, precision, under_lowestbin, above_highestbin):
    ### Attributes the correct bin ........................
    ### Function that, based on the cutpoints, seeks the lowest precision necessary to have meaningful bins
    ###  e.g. (5.5,5.5] ==> (5.51,5.54]
    ### Attributes those bins to each value of x, to achieve a binned version of x   
        
        # Store unique cutpoints (e.g. from 1,3,3,5 to 1,3,5) to avoid inconsistensies when bin-label making
        # Indeed, bins [...,1], (1,3], (3,3], (3,5], (5,...] do not make much sense
        # While, bins  [...,1], (1,3],        (3,5], (5,...] do make sense
        unique_cutpoints = cutpoints.unique()
        # If there are only 2 unique cutpoints (and thus only one bin will be returned), 
        # keep original values and code missings as 'Missing'
        if len(unique_cutpoints) <= 2:
            cutpoints = None
            x_binned = pd.Series(x)
            x_binned[x_binned.isnull()] = 'Missing'
            info = (var.name+": Only one resulting bin, keeping original values instead")
            return x_binned, cutpoints, info
        # Store info on whether or not the number of resulting bins equals the desired number of bins
        elif len(unique_cutpoints) < len(cutpoints):
            info = (var.name+": Resulting # bins < whished # bins")
        else:
            info = (var.name+": Resulting # bins as desired")
        # Finally, recode the cutpoints (which can have doubles) as the unique cutpoints
        cutpoints = unique_cutpoints
        
        # Store missing values in the variable as a mask, and create a flag to test if there are any missing in the variable
        na_mask = np.isnan(x)
        has_nas = na_mask.any()
        # Attribute to every x-value the index of the cutpoint (from the sorted cutpoint list) which is equal or higher than
        # the x-value, effectively encompasing that x-value.
        # e.g. for x=6 and for sorted_cutpoint_list=[0,3,5,8,...] the resulting_index=3    
        ids = cutpoints.searchsorted(x, side='left')
        # x-values equal to the lowest cutpoint will recieve a ids value of 0
        # but our code to attribute bins to x-values based on ids (see end of this subfunction) requires a min. value of 1
        ids[x == cutpoints[0]] = 1
        # Idem as previous: x-values below the lowest cutpoint should recieve a min. value of 1
        if under_lowestbin.any():
            ids[under_lowestbin] = 1
        # Similar as previous: x-values above the highest cutpoint should recieve the max. allowed ids
        if above_highestbin.any():
            max_ids_allowed = ids[(above_highestbin == False) & (na_mask==False)].max()
            ids[above_highestbin] = max_ids_allowed
        # Maximal ids can now be defined if we neglect ids of missing values
        max_ids = ids[na_mask==False].max()
        
        # Based on the cutpoints create bin-labels
        # Iteratively go through each precision (= number of decimals) until meaningful bins are formed
        # If theoretical bin is ]5.51689,5.83654] we will prefer ]5.5,5.8] as output bin
        increases = 0
        original_precision = precision
        while True:
            try:
                bins = _format_bins(cutpoints, precision)
            except ValueError:
                increases += 1
                precision += 1
                #if increases >= 5:
                    #warnings.warn("Modifying precision from "+str(original_precision)+" to "+str(precision)+" to achieve discretization")
                    #print("Modifying precision from "+str(original_precision)+" to "+str(precision)+" to achieve discretization")
            else:
                break
        
        # Make array of bins to allow vector-like attribution
        bins = np.asarray(bins, dtype=object)
        # If x has nas: for each na-value, set the ids-value to max_ids+1
        # this will allow na-values to be attributed the highest bin which we define right below
        if has_nas:
            np.putmask(ids, na_mask, max_ids+1)
        # The highest bin is defined as 'Missing'
        bins = np.append(bins,'Missing')
        # ids-1 is used as index in the bin-labels list to attribute a bin-label to each x. Example:
        # x=6   sorted_cutpoint_list=[0,3,5,8,...]   ids=3   levels=[[0,3],(3,5],(5,8],...]
        # The correct bin level for x is (5,8] which has index 2 which is equal to the ids-1
        x_binned = bins[ids-1]
        return x_binned, cutpoints, info
        

    def _format_bins(cutpoints, prec):
    # Based on the quantile list create bins. Raise error if values are similar within one bin.
    # On error _binnedx_from_cutpoints will increase precision
        
        fmt = lambda v: _format_label(v, precision=prec)
        bins = []
        for a, b in zip(cutpoints, cutpoints[1:]):
            fa, fb = fmt(a), fmt(b)
            
            if a != b and fa == fb:
                raise ValueError('precision too low')
                
            formatted = '(%s, %s]' % (fa, fb)
            bins.append(formatted)
        
        bins[0] = '[...,' + bins[0].split(",")[-1]
        bins[-1] = bins[-1].split(",")[0] + ',...]'
        return bins


    def _format_label(x, precision):
    # For a specific precision, returns the value formatted with the appropriate amount of numbers after comma and correct brackets
    
        if isinstance(x,float):
            frac, whole = np.modf(x)
            sgn = '-' if x < 0 else ''
            whole = abs(whole)
            if frac != 0.0:
                val = '{0:.{1}f}'.format(frac, precision)
                val = _trim_zeros(val)
                if '.' in val:
                    return sgn + '.'.join(('%d' % whole, val.split('.')[1]))
                else: 
                    if '0' in val:
                        return sgn + '%0.f' % whole
                    else:
                        return sgn + '%0.f' % (whole+1)
            else:
                return sgn + '%0.f' % whole
        else:
            return str(x)


    def _trim_zeros(x):
    # Removes unnecessary zeros and commas
        while len(x) > 1 and x[-1] == '0':
            x = x[:-1]
        if len(x) > 1 and x[-1] == '.':
            x = x[:-1]
        return x


    x_binned, cutpoints, info = _binnedx_from_cutpoints(x, cutpoints, precision=precision, under_lowestbin=under_lowestbin, above_highestbin=above_highestbin)
    x_binned = pd.Series(x_binned, index=series_index, name="B_"+name)
    return x_binned, cutpoints, info


# # WIP

# for n in numeric_headers:
#     result = eqfreq(var=df_in[n]
#                     ,train=df_in["PARTITION"]=="train"
#                     ,autobins=True
#                     ,nbins=int(df_settings.discretization_nbins)
#                     ,precision=0
#                     ,twobins=True
#                     ,catchLarge=True)
#     print(n)
#     print(result[0].unique())
#     print('\n')

# # /WIP

# Apply function to continuous variables

# In[17]:

tic = time.time()
# We loop only through the numeric variables
for n in numeric_headers:
    # Perform the equifrequency function
    result = eqfreq(var=df_in[n]
                    ,train=df_in["PARTITION"]=="train"
                    ,autobins=True
                    ,nbins=int(df_settings.discretization_nbins)
                    ,precision=0
                    ,twobins=True
                    ,catchLarge=False) # TRUE OPTION STILL PRODUCES ERROR IN SORTNUMERIC function AND SCORING procedure !!!!!!!!!
    df_out = pd.concat([df_out,result[0]], axis=1)
    log.append(result[2])
toc = time.time()
log.append("Discretisation: "+str(toc-tic)+" sec"+"\n")


# ---

# ### Preprocessing of categorical variables

# Function for labeling missing/empty values

# In[18]:

# Check which values of a var are empty strings or null values
def maskmissing(var):
    # Check if values are null
    crit1 = var.isnull()
    # Check if values are empty strings
    modvar = pd.Series([str(value).strip() for value in var])
    crit2 = modvar==pd.Series(['']*len(var))
    #crit2 = var==pd.Series(['']*len(var))
    #crit3 = var==pd.Series([' ']*len(var))
    return crit1 | crit2 


# Regrouping Function for nominal/ordinal variables

# In[19]:

# Regrouping function for categorical variables
# Each group is tested with a chi² for relevant incidence differences in comparison to a rest-group
# The rest group has the size of the remaining groups and an 'overall average incidence' (if dummy=True) or 
# 'remaining groups average incidence' (if dummy=False)
# Groups with a pvalue above the threshold are relabled to a single group

def regroup(var,target,train,pval_thresh=0.01,dummy=True,keep='Missing',rename='Other'):
    
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


# Apply function to nominal/ordinal variables

# In[20]:

tic = time.time()
# We loop only through the categorical variables
for h in object_headers:
    # We label missing and empty values for categorical variables as 'Missing'
    # Note the interaction with the 'keep' parameter of the regroup function.
    mask = maskmissing(df_in[h])
    df_in.loc[mask,h]='Missing'
    # Perform regrouping function
    result = regroup(var=df_in[h]
                     ,target=df_in.loc[:,'TARGET']
                     ,train=df_in.PARTITION=='train'
                     ,pval_thresh=float(df_settings.regrouping_signif)
                     ,dummy=True
                     ,keep='Missing'
                     ,rename='Non-significants')
    df_out = pd.concat([df_out,result[0]],axis=1)
    log.append(result[1])
toc = time.time()
log.append("Regrouping: "+str(toc-tic)+" sec"+"\n")


# ---

# ### Preprocessing of boolean variables

# Defining Function to pass variables as is

# In[21]:

# We could just rename them or put them with the regoup function, but for now let's keep consistent with the other functions
def passvar(var):
    var_pass = var.copy()
    var_pass.name = "B_"+var.name
    info = ("Passing "+var.name)
    return var_pass, info


# Executing function

# In[22]:

tic = time.time()
# We loop only through the boolean variables
for b in bool_headers:
    # We label missing and empty values for boolean variables as 'Missing'
    mask = maskmissing(df_in[b])
    df_in.loc[mask,b]='Missing'
    # Perform the passvar function
    result = passvar(var=df_in[b])
    df_out = pd.concat([df_out,result[0]],axis=1)
    log.append(result[1])
toc = time.time()
log.append("Passing: "+str(toc-tic)+" sec"+"\n")


# ---

# ### Incidence Replacement

# Function for incidence replacement

# In[23]:

def increp(b_var, target, train):    
    #get variable name
    name = b_var.name
    #get overall incidence 
    incidence_mean = target[train].mean()
    #get incidence per group
    incidences = target[train].groupby(b_var).mean()
    #construct dataframe with incidences
    idf = pd.DataFrame(incidences).reset_index()
    #get values that are in the data but not in the labels
    bin_labels = incidences.index
    newgroups = list(set(b_var.unique()) ^ set(bin_labels))
    #if newgroups, add mean incidence to incidence dataframe for each new group
    if len(newgroups)>0:
        #make dataframe:
        ngdf = pd.DataFrame(newgroups)
        ngdf.columns = [name]
        ngdf["TARGET"] = incidence_mean
        #dataframe with incidences:    
        idf = idf.append(ngdf)
    #dataframe with the variable
    vdf = pd.DataFrame(b_var)
    #discretized variable by merge
    d_var = pd.merge(vdf,idf,how='left',on=name)["TARGET"]
    return pd.Series(d_var, name="D_"+name[2:]) 


# Apply function for incidence replacement

# In[24]:

# We define the columns destined for incidence replacement
headers_for_incidrep = [h for h in df_out.columns if ((h not in ['ID','TARGET','PARTITION']) & (h[:2]=="B_"))]


# In[25]:

tic = time.time()
# We loop only through the columns destined for incidence replacement
for n in headers_for_incidrep:
    # Perform increp function
    result = increp(b_var=df_out[n]
                    ,target=df_out.TARGET
                    ,train=df_out.PARTITION=="train")
    df_out = pd.concat([df_out,result], axis=1)
    log.append(n+ " processed")
toc = time.time()
log.append("Incidence replacement: "+str(toc-tic)+" sec"+"\n")


# ---

# ### Calculate AUCS

# Function for auc calculation

# In[26]:

def getauc(var, target, partition):     
    
    y = np.array(target[partition])
    pred = np.array(var[partition])
    pred = pred.astype(np.float64)
    fpr, tpr, thresholds = metrics.roc_curve(y,pred, pos_label=1)
    
    return metrics.auc(fpr, tpr)


# Applying function for auc calculation

# In[27]:

# We define the columns for which an AUC score should be computed
headers_for_auc = [h for h in df_out.columns if ((h not in ['ID','TARGET','PARTITION']) & (h[:2]=="D_"))]


# In[28]:

auc_list_all = []
parts = ["train","selection"]
tic = time.time()
# We loop only through those columns for which an AUC score should be computed
for header in headers_for_auc:
    auc_list_var = [header[2:]]
    # We loop through the two sets ('train' and 'selection') for which an AUC score is needed
    for part in parts:
        # Perform getauc function
        auc_value = getauc(var=df_out[header]
                           ,target=df_out.TARGET
                           ,partition=df_out.PARTITION==part)
        auc_list_var.append(auc_value.round(2)) #We round auc values to 2 decimals
    auc_list_all.append(auc_list_var)
    log.append(header + " processed")
# We create a supplementary dataframe destined for Cobra input  
df_auc = pd.DataFrame(auc_list_all,columns=['variable','AUC train','AUC test'])
toc = time.time()
log.append("Auc: "+str(toc-tic)+" sec"+"\n")


# ---

# ### Preselection

# In[29]:

tic = time.time()
# We identify those variables for which the AUC score is above the user-defined threshold
auc_thresh = df_auc.loc[:,'AUC test'] > float(df_settings.preselection_auc)
# We identify those variables for which the AUC score difference between 'train' and 'selection' is within the user-defined ratio
auc_overtrain = (df_auc.loc[:,'AUC train']*100 - df_auc.loc[:,'AUC test']*100) < float(df_settings.preselection_overtrain)
# Only those variables passing the 2 criteria above are preselected
preselect = auc_thresh & auc_overtrain


# In[30]:

# We create a supplementary dataframe destined for Cobra input  
df_variable_selections = pd.DataFrame({'variable':df_auc.variable
                                      ,'preselect':preselect.astype(int)
                                      ,'Default':np.zeros(len(preselect)).astype(int)
                                      ,'Alternative 1':np.zeros(len(preselect)).astype(int)
                                      ,'Alternative 2':np.zeros(len(preselect)).astype(int)
                                      ,'Alternative 3':np.zeros(len(preselect)).astype(int)
                                      ,'Alternative 4':np.zeros(len(preselect)).astype(int)
                                      ,'Alternative 5':np.zeros(len(preselect)).astype(int)}
                                     ,columns=['variable'
                                               ,'preselect'
                                               ,'Default'
                                               ,'Alternative 1'
                                               ,'Alternative 2'
                                               ,'Alternative 3'
                                               ,'Alternative 4'
                                               ,'Alternative 5'])


# In[31]:

for i,var in enumerate(df_variable_selections.variable):
    log.append(var+" "+np.array(['passed','filtered'])[df_variable_selections.preselect][i])
toc = time.time()
log.append("Preselection: "+str(toc-tic)+" sec"+"\n")


# ---

# ### Calculate Correlations

# In[32]:

# We define the columns for which a correlation score should be computed
headers_for_corr = [h for h in df_out.columns if ((h not in ['ID','TARGET','PARTITION']) & (h[:2]=="D_"))]


# In[33]:

train = df_out.PARTITION=="train"
tic = time.time()
dataforcorr = np.transpose(np.matrix(df_out.loc[train,headers_for_corr],dtype=float))
with np.errstate(invalid='ignore', divide='ignore'):
    mat_corr = np.corrcoef(dataforcorr)
toc = time.time()
log.append("Correlations: "+str(toc-tic)+" sec"+"\n")


# In[34]:

df_corr = pd.DataFrame(mat_corr)
df_corr.columns = headers_for_corr
df_corr.index = headers_for_corr
df_corr.fillna(0, inplace=True)

# ---

# ### Export files

# Table of all Auc values

# In[35]:

auc_path = root+'/data/univariate/aucs.csv'
df_auc = df_auc.sort_values(by=['AUC test','AUC train'], ascending=False).reset_index(drop=True)
df_auc.to_csv(path_or_buf=auc_path
              ,sep=';'
              ,index=False
              ,encoding='utf-8'
              ,line_terminator='\n')


# Tables of Incidences & Correlations per variable

# In[36]:

# Function for sorting cont.variables, whether or not they have undergone discritization
def sortnumeric(dataframe):
    
    lowestnumber = 0
    # If the variable was discretisized
    if '[...' in [str(l)[:4] for l in dataframe.group.values]:
        unsorted_labels = dataframe.group.values
        label_items=[]
        for label in unsorted_labels:
            # For each bin label, retain the first value
            label_items.append(label.split(",")[0].strip("[").strip("("))
        label_items=np.asarray(label_items)
        # Special cases that are not numeric are given numbers
        lowestnumber = label_items[(label_items!="...")&(label_items!="Missing")].astype('float64').min()
        label_items[label_items=='...']= lowestnumber-1
        label_items[label_items=='Missing']= lowestnumber-2
        # argsort based on the numbers
        rank = label_items.astype('float64').argsort()
        return rank
    
    # If the variable wasn't discretisized, simply argsort on the numbers
    else:
        label_items = dataframe.group.values
        if len(label_items)>1:
            lowestnumber = label_items[label_items.astype('O')!="Missing"].astype('float64').min()
        label_items[label_items.astype('O')=='Missing']= lowestnumber-2
        rank = label_items.astype('float64').argsort()
        return rank


# In[37]:

# Function for sorting cont.variables, whether or not they have undergone discritization
def sortnumeric_old(dataframe):
    
    # If the variable was discretisized
    if dataframe.group.dtype=='object': #or# if np.array([str(unsorted_labels[i])[0] in ["[","(","M"] for i in range(0,len(unsorted_labels))]).all():
        unsorted_labels = dataframe.group.values
        label_items=[]
        for label in unsorted_labels:
            # For each bin label, retain the first value
            label_items.append(label.split(",")[0].strip("[").strip("("))
        label_items=np.asarray(label_items)
        # Special cases that are not numeric are given numbers
        lowestnumber = label_items[(label_items!="...")&(label_items!="Missing")].astype('float64').min()
        label_items[label_items=='...']= lowestnumber-1
        label_items[label_items=='Missing']= lowestnumber-2
        # argsort based on the numbers
        rank = label_items.astype('float64').argsort()
        return rank
    
    # If the variable wasn't discretisized, simply argsort on the numbers
    else:
        rank = dataframe.group.values.argsort()
        return rank


# In[38]:

# Function for sorting cat. variables
def sortobject(dataframe):
    # Sort dataframe on increasing incidence values
    unsorted_incidences = dataframe.incidence.values
    rank = unsorted_incidences.argsort()
    return rank


# In[39]:

n_decimals = 2
average = round(df_out.TARGET[df_out.PARTITION=="train"].mean(),n_decimals)

headers_to_output = list(df_auc['variable'])
for i,varname in enumerate(headers_to_output):
    b_varname = 'B_'+varname
    d_varname ='D_'+varname
    #INCIDENCE CSV's
    incidence_path = root+"/data/univariate/incidence_"+str(varname)+".csv"
    groups_and_incidences = df_out.TARGET[df_out.PARTITION=='train'].groupby(df_out[b_varname]).mean()
    n_groups= len(groups_and_incidences)
    group = groups_and_incidences.index
    incidence = groups_and_incidences.values.round(n_decimals)
    size = df_out.TARGET[df_out.PARTITION=='train'].groupby(df_out[b_varname]).size().astype(float).values
    df_incidence = pd.DataFrame( {'group':group
                                  ,'incidence':incidence
                                  ,'size':size
                                  ,'average':average}
                                ,columns=['group','incidence','size','average'])
    if varname in numeric_headers:
        df_incidence = df_incidence.iloc[sortnumeric(df_incidence),:]
    elif varname in object_headers:
        df_incidence = df_incidence.iloc[sortobject(df_incidence),:]
    else:
        a=1
        #df_incidence = df_incidence.iloc[sortother(df_incidence),:]
    df_incidence.to_csv(path_or_buf=incidence_path
                        ,sep=';'
                        ,index=False
                        ,encoding='utf-8'
                        ,line_terminator='\n') #quoting=csv.QUOTE_NONNUMERIC
    
    #CORRELATION CSV's
    correlation_path = root+"/data/univariate/correlations_"+str(varname)+".csv"
    Variable = [v.strip("D_") for v in df_corr[d_varname].index]
    Correlation = abs(df_corr[d_varname].values).round(n_decimals)
    Sign = np.array(["+","-"])[(df_corr[d_varname].values<0).astype(int)]
    AUC = np.array([df_auc.loc[df_auc['variable']== v,'AUC test'].values[0] for v in Variable]).round(n_decimals)
    df_correlation = pd.DataFrame({"Variable":Variable
                                   ,"Correlation":Correlation
                                   ,"Sign":Sign
                                   ,"AUC": AUC}
                                  ,columns=["Variable","Correlation","Sign","AUC"]) 
    df_correlation.sort_values(by='Correlation', ascending=False, inplace=True)
    df_correlation = df_correlation.loc[df_correlation.Variable!=varname,:]
    df_correlation.to_csv(path_or_buf=correlation_path
                          ,sep=';'
                          ,index=False
                          ,encoding='utf-8'
                          ,line_terminator='\n') # quoting=csv.QUOTE_NONNUMERIC


# Variable Preselections

# In[ ]:

selections_path = root+'/data/univariate/variable_selections.csv'
df_variable_selections.to_csv(path_or_buf=selections_path
                              ,sep=';'
                              ,index=False
                              ,encoding='utf-8'
                              ,line_terminator='\n')


# Result dataframe for Modeling input

# In[ ]:

out_path = root+"/data/univariate/df_univariate.csv"
df_out.to_csv(path_or_buf=out_path, sep=';', index=False, encoding='utf-8', line_terminator='\n', quoting=csv.QUOTE_NONNUMERIC)


# Modeltab reset

# In[ ]:

# Generate modeltab info
filename = root+"/data/univariate/modeltab_info.csv"
with open(filename, 'w') as csvfile:
    write=csv.writer(csvfile, delimiter =';')
    write.writerow(["key","value"])
    write.writerow(["run","Default"])
    write.writerow(["new","Alternative 1"])
    write.writerow(["new_template","Default"])
    write.writerow(["champ","Default"])
    write.writerow(["score","Default"])


# Log messages

# In[ ]:

log.append("-- Univariate analysis completed --"+"\n")


# In[ ]:

log_file = open(root+'/python/univariate.log','w')
log_file.write('\n'.join(log))
log_file.close()


# ---

# ### Stop script

# In[ ]:

print("ok")


# ---
