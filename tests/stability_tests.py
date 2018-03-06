''' 
======================================================================================================================
----------------------------------------------------  STABILITY TESTING  ---------------------------------------------
======================================================================================================================
TEST stability of partitioning and binning
-> binning IS stable, but partitioning is not
-> as a result, number of binns and their size is not stable
-> therefore, incidence replacement will differ
-> which means that the logit will be trained on different data, giving different coefficients and AUC
-> which leads to instability 
   i) forward selection stops because there are no positive coefs
   ii) it has an effect on AUC - it can be lower so we will drop the variable even if it is useful
-> I found this behavior only for one variables, but I belive it will be case of more
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
pd.set_option("display.max_columns",50)

data_path = 'C:/Local/pers/Documents/GitHub/COBRA/datasets/data.csv'
data_types_path = 'C:/Local/pers/Documents/GitHub/COBRA/datasets/data_types.csv'



def __eqfreq(var, train, autobins=True, nbins=10, precision=0, twobins=True, catchLarge=True):
    '''
    Special method for binning continuous variables into bins
    ----------------------------------------------------
    var: input pd.Serie with continuous columns
    train: mask with rows which belongs to train
    autobins: adapts number of bins
    nbins: number of bins
    precision: precision to form meaningful bins
    twobins: if only two bins are found, iterate to find more
    catchLarge: check when groups are too big
    ---------------------------------------------------- 
    - This function is a reworked version of pd.qcut to satisfy our particular needs
    - Takes for var a continuous pd.Series as input and returns a pd.Series with bin-labels (e.g. [4,6[ )
    - Train takes a series/list of booleans (note: we define bins based on the training set)
    - Autobins reduces the number of bins (starting from nbins) as a function of the number of missings
    - Nbins is the wished number of bins
    - Precision=0 results in integer bin-labels if possible
    - twobins=True forces the function to output at least two bins
    - catchLarge tests if some groups (or missing group) are very large, and if so catches and outputs two groups
    - note: catchLarge makes twobins irrelevant
    '''

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

def __increp(b_var, target, train):    
    '''
    Method for incidence replacement
    Returns replaced pd.Serie
    ----------------------------------------------------
    b_var: input pd.Serie to be replaced
    target: pd.Serie with target variable
    train: pd.Serie with parition variable
    ---------------------------------------------------- 
    '''
    
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




#
# LOAD CSV
#
df = pd.read_csv(data_path, header=0, sep=None, engine='python')

#
# PREPARE DATA
#
clmn = 'scont_1'
df_prep = df[['TARGET',clmn]]

#
# ADD PARTITION
#
import math

np.random.seed(0)

_partitioning_settings = {'train':0.5,
                          'selection':0.3, 
                          'validation':0.2}


#
# BINNING
#
df_simulation = pd.DataFrame(None,columns=[
                                          'iteration',
                                          'length',
                                          'coef',
                                          'AUC'
                                          ])
row = 0
for i in range(1,51):
    #PARTITION
    df_prep = df_prep.iloc[np.random.permutation(len(df_prep))].sort_values(by='TARGET', ascending=False).reset_index(drop=True)
    partition = []
    sorted_target=df_prep['TARGET'] #Just the target since it is allready sorted (see above)
    for target in [sorted_target.iloc[0],sorted_target.iloc[-1]]:
        target_length = (sorted_target==target).sum()
        
        for part, size in _partitioning_settings.items():
            partition.extend([part]*math.ceil(target_length*size))
            
    df_prep["PARTITION"] = partition[:len(df_prep)]
    
    #Binns
    result = __eqfreq(var=df_prep[clmn],
                      train=df_prep["PARTITION"]=="train",
                      autobins=True,
                      	nbins=5,
                      precision=0,
                      twobins=True,
                      # TRUE OPTION STILL PRODUCES ERROR IN SORTNUMERIC function AND SCORING procedure !!!!!!!!!
                     catchLarge=False)
    
    bin_serie = pd.Series(result[0])
    # uncommemt this to see the counts - they change!
    #print(bin_serie.groupby(bin_serie).count())
    
    #REPLACE INCIDENCE
    inc_rep = __increp(b_var=bin_serie,
                       target=df_prep['TARGET'],
                       train=df_prep['PARTITION']=="train") 
    
    df_prep['D_'+clmn] = inc_rep
   
    #PREDICT
    y_train = df_prep['TARGET'][df_prep['PARTITION'] == 'train'].as_matrix()
    x_train = df_prep['D_'+clmn][df_prep['PARTITION'] == 'train'].as_matrix().reshape(-1,1)
    
    logit = LogisticRegression(fit_intercept=True, C=1e9, solver = 'liblinear')
    logit.fit(y=y_train, X=x_train)
    y_pred_train = logit.predict_proba(x_train)
    
    AUC_train = metrics.roc_auc_score(y_true=y_train, y_score=y_pred_train[:,1])
    
    coefs = logit.coef_[0]
    
    df_simulation.loc[row] = [
                             i,
                             len(np.unique(result[0])),
                             coefs[0],
                             np.round(AUC_train,3)
                             ]
    row +=1
    
    print('ITERATION {}, lenght: {}, coef: {}, AUC: {}.'.format(i, len(np.unique(result[0])),coefs,np.round(AUC_train,3)))
    
print('Std. Dev of coefs is: {}.'.format(df_simulation['coef'].std(axis=0)))
print('Mean of coefs is: {}.'.format(df_simulation['coef'].mean(axis=0)))

print('Std. Dev of AUC is: {}.'.format(df_simulation['AUC'].std(axis=0)))
print('Mean of AUC is: {}.'.format(df_simulation['AUC'].mean(axis=0)))




''' 
df_transformed.groupby('D_scont_1')['D_scont_1'].count()
df_transformed.groupby('B_scont_1')['B_scont_1'].count()

res = pd.Series(result[0]) 
res.groupby(res).count()
'''


















