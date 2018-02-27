
# coding: utf-8

# # Modeling

# ### General Imports

# In[1]:

import time
import math
import random
import csv
import os


# In[2]:

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


# In[3]:

from scipy import stats
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from itertools import chain 


# ---

# ### Miscellaneous

# In[4]:

log = []


# In[5]:

# When code is in script, we define the path of the script's parent folder location as the root directory
# From this root we can travel to the relevant folders with minimal adjustment
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

# Read-in univariate output with asssumed ID, TARGET, PARTITION and D_VARS

# In[7]:

df_univariate_path = root+"/data/univariate/df_univariate.csv"
df_in = pd.read_csv(df_univariate_path, sep=";")


# Reference X and Y for each partition individually

# In[8]:

dvars = [n for n in df_in.columns if n[:2] == 'D_']


# In[9]:

mask_train = df_in.PARTITION=="train"
mask_selection = df_in.PARTITION=="selection"
mask_validation = df_in.PARTITION=="validation"


# In[10]:

y_train = df_in.loc[mask_train,'TARGET']
y_selection = df_in.loc[mask_selection,'TARGET']
y_validation = df_in.loc[mask_validation,'TARGET']


# In[11]:

x_train = df_in.loc[mask_train,dvars]
x_selection = df_in.loc[mask_selection,dvars]
x_validation = df_in.loc[mask_validation,dvars]


# Analysis settings

# In[12]:

df_settings = pd.read_csv(root+'/python/analysis_settings.csv', sep=',', index_col=0, header=None).T


# Modeltab info

# In[13]:

df_modeltab = pd.read_csv(root+'/data/univariate/modeltab_info.csv',sep=';', index_col=0, header=None).T
modelrun = df_modeltab.run[1]


# Variable selections

# In[14]:

df_selections = pd.read_csv(root+'/data/univariate/variable_selections.csv',sep=';')


# ---

# ### Model making and recording

# Define functions

# In[15]:

# Function to make logistic model on a predefined set of predictors + compute train AUC of resulting model 
def processSubset(predictors_subset):
    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics
    # Fit model on predictors_subset and retrieve performance metric
    model = LogisticRegression(fit_intercept=True, C=1e9, solver = 'liblinear')
    modelfit = model.fit(y=y_train, X=x_train[predictors_subset])
    # Position of the TARGET==1 class
    pos = [i for i,h in enumerate(modelfit.classes_) if h==1]
    # Prediction probabilities for the TARGET==1
    y_pred = modelfit.predict_proba(x_train[predictors_subset])[:,pos]
    auc = metrics.roc_auc_score(y_true=y_train, y_score=y_pred)
    return {"modelfit":modelfit,"auc":auc,"predictor_names":predictors_subset,"predictor_lastadd":predictors_subset[-1]}


# In[16]:

# Function for computing AUC of all sets (train, selection & validation)
def getAuc(df_without_auc):
    import pandas as pd
    from sklearn import metrics
    df_with_auc = df_without_auc[:]
    for x,y,part in [(x_train,y_train,'train'),
                    (x_selection,y_selection,'selection'),
                    (x_validation,y_validation,'validation')]:
        pos = [i for i,h in enumerate(df_without_auc.modelfit.classes_) if h==1]
        y_pred = df_without_auc.modelfit.predict_proba(x[df_without_auc['predictor_names']])[:,pos]
        df_with_auc["auc_"+part] = metrics.roc_auc_score(y_true=y, y_score=y_pred)
        df_with_auc["pred_"+part] = y_pred
    return(df_with_auc)


# In[17]:

# Forward selection function that uses processSubset and getAuc
def forward(current_predictors, pool_predictors, positive_only=True):
    import pandas as pd
    import numpy as np
    tic = time.time()
    
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
            results.append(processSubset(current_predictors+[p]))
        except:
            errorcount += 1 
    models = pd.DataFrame(results)
    
    # If we require all coefficients to be positive...
    if positive_only:
        #Create a flag for each submodel to test if all coefficients are positive 
        all_positive = pd.Series(None, index=models.index)
        for i in range(0,len(models)):
            all_positive[i] = (models.modelfit[i].coef_ >= 0 ).all()
            
        # if no model exist with only positive coefficients raise error we can easily identify as normal
        if (all_positive==0).all():
            raise ValueError("No models with only positive coefficients","NormalStop")
            
        #Choose model with best performance and only positive coefficients
        best_model = models.loc[models[all_positive==1].auc.argmax()]
        best_model = getAuc(best_model)
        
    # If we don't require all coefficients to be positive...   
    else:
        #Choose model with best performance
        best_model = models.loc[models.auc.argmax()]
        best_model = getAuc(best_model)

    
    tac = time.time()
    info = ("Processed "
            + str(models.shape[0])
            + " models on "
            + str(len(current_predictors)+1) 
            + " predictors in " 
            + str(round(tac-tic,2)) 
            +" sec with " 
            + str(errorcount) 
            +" errors")
    
    return best_model, info


# Create recipient vars

# In[18]:

best_models = pd.DataFrame(columns=["modelfit",
                                    "predictor_names",
                                    "predictor_lastadd",
                                    "auc_train",
                                    "auc_selection",
                                    "auc_validation",
                                    "pred_train",
                                    "pred_selection",
                                    "pred_validation"])
predictors = []


# Define number of steps depending on settings and total number of predictors

# In[19]:

step_setting = int(df_settings.modeling_nsteps)
n_steps = min(step_setting,len(x_train.columns))


# Define which variables to pass, force and filter

# In[20]:

mask_pass = (df_selections.preselect == 1) & (df_selections[modelrun]==0)
varname_list_pass = 'D_'+df_selections.loc[mask_pass,'variable']
length_pass = len(varname_list_pass)

mask_force = (df_selections.preselect == 1) & (df_selections[modelrun]==1)
varname_list_force = 'D_'+df_selections.loc[mask_force,'variable']
length_force = len(varname_list_force)


# Execute forward modeling process

# In[21]:

tic = time.time()
use_predictors = varname_list_force #x_train.columns
for i in range(1,n_steps+1):
    try:
        # Use predictors to be forced first. Once through the list, append the remaining variables to be passed.
        use_predictors = varname_list_force.append(varname_list_pass[[i>length_force]*length_pass]).reset_index(drop=True)
        result = forward(current_predictors=predictors
                         ,pool_predictors= use_predictors
                         ,positive_only=True)
        best_models.loc[i] = result[0]
        predictors = best_models.loc[i].predictor_names
        log.append(result[1])
    except Exception as e:
        # Normal errors (i.e. no more predictors to be used / no models with only positive coefficients)
        if e.args[-1]=='NormalStop':
            log.append("Stopped modeling at "+str(i)+" predictors: "+ e.args[-2])
        # Other unknown errors
        else:
            log.append("Stopped modeling at "+str(i)+" predictors: unknown error")
        break
toc = time.time()
log.append("Forward selection modeling: " + str(round((toc-tic)/60,0)) + " min"+"\n")


# ---

# ### Optimal model criterion

# Define functions

# In[22]:

def comparefit(p,g=2):
    # We fit a second degree (g=2) polyline through our auccurve 
    # This serves as a starting base for finding our optimal stopping point
    import numpy as np
    import pandas as pd
    z = np.polyfit(p.index, p, g)
    f = np.poly1d(z)
    y_new = f(p.index)
    return pd.Series(y_new,index=p.index)


# In[23]:

def slopepoint(p,p_fit,thresh_ratio=0.2):
    # We take the polyline from comparefit and look for the point of which the slope lies just below some percentage of the max. slope
    slopes = [p_fit[i+1]-p_fit[i] for i in range(1,len(p_fit))]
    slopes = pd.Series(slopes, index=range(1,len(p_fit)))
    thresh = slopes.max()*thresh_ratio
    p_best_index = (slopes[slopes>thresh])[-1:].index
    p_best = p.loc[p_best_index]
    return p_best


# In[24]:

def moveright(p,p_fit,p_best,n_steps=5,dampening=0.01):
    # We look nsteps right on the polyline (starting from the slopepoint) and take the point with largest difference with real line
    # We move to that point if that difference is larger than some multiplication of the difference at the slopepoint
    # That multiplication gets larger as current the current difference gets smaller with a certain amount of dampening. 
    # The rationale behind this is as follows: 
    #  if the current difference is already large than the larger difference will definitely be noteworthy
    #  if however the current difference is near zero than there needs to be much larger difference to be noteworthy
    in_index = p_best.index.values[0]
    lower = (in_index-1)
    upper = (in_index+n_steps-1)
    p_diff = p[lower:upper]-p_fit[lower:upper]
    out_index = p_diff.argmax()
    factor = 1/abs(p_diff[in_index])
    if (p_diff[out_index]>p_diff[in_index]+(abs(p_diff[in_index])*factor*dampening)):
        p_best_new = pd.Series(p[out_index],index=[out_index])
    else:
        p_best_new = p_best
    return p_best_new


# In[25]:

def moveleft(p,p_fit,p_best,rangeloss=0.1, diffshare=0.8): #diff_min=0.005):
    # Starting from whatever point we end up with (either the slopepoint or a move to the right)
    # We look left on the polyline and take the point for which the real line is largest (current point included)
    # We move left if we stay within [a specific % loss of range] AND [a minimum % of current difference]
    # i.e. we don't won't to go to low compared to the overall real line
    #      and we don't won't to move to a point that does not make a significant increase in AUC (i.e. difference between polyline and real line)
    p_left = p[:p_best.index.values[0]]
    p_best = p_left[p_left==p_left.max()]
    p_diff = p-p_fit
    p_range = p.max()-p.min()
    s = p[(p >= p_best.values[0]-(rangeloss*p_range)) 
          & (p.index <= p_best.index.values[0]) 
          & (p_diff>=diffshare*p_diff[p_left.index[-1]])
         ]
    p_best_new = s[s.index == s.index.values.min()]
    return p_best_new


# Execute functions

# In[26]:

points = best_models.auc_selection
points_fit = comparefit(p=points, g=2)
points_slope = slopepoint(p=points, p_fit=points_fit, thresh_ratio=0.2)
points_right = moveright(p=points, p_fit=points_fit, p_best=points_slope, n_steps=5, dampening=0.01)
points_left = moveleft(p=points, p_fit=points_fit, p_best=points_right, rangeloss=0.1, diffshare=0.8)

optimal_nvars = points_left.index.values[0]


# Inspect

# %matplotlib inline
# from pylab import rcParams
# rcParams['figure.figsize'] = 10, 5
# 
# plt.plot( points.index      , points          , color="blue")
# plt.plot( points_fit.index  , points_fit      , color="red")
# plt.plot( points_slope.index, points_slope,'o', color="lightgreen", markersize=12)
# plt.plot( points_right.index, points_right,'o', color="black"     , markersize=8)
# plt.plot( points_left.index , points_left ,'o', color="gold"      ,  markersize=4)
# 
# axes = plt.gca()
# axes.set_ylim([0.45,1])
# plt.show()

# ---

# ### Cumulative gains/response

# Define functions

# In[28]:

# Compute cumulative response/gains
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


# Execute functions

# In[29]:

cresp_all = [None]
cgains_all = [None]
for i in range(1,len(best_models)+1):
    out = cumulatives(y=y_selection
                      ,yhat=best_models.pred_selection[i][:,0]
                      ,perc_as_int=True
                      ,dec=2)
    cresp_all.append(out[0]) 
    cgains_all.append(out[1])


# Inspect

# %matplotlib inline
# from pylab import rcParams
# rcParams['figure.figsize'] = 10, 5
# 
# cmap = plt.get_cmap('hot')
# colors = [cmap(i) for i in np.linspace(0, 1, n_steps)]
# for i in range(1,len(best_models)):
#     plt.plot(range(1,101), cresp_all[i], color=colors[i-1])
# plt.plot(range(1,101), cresp_all[-1], color="black")
#             
# axes = plt.gca()
# axes.set_ylim([0,max(max(l) for l in np.array(cresp_all)[1:])])
# plt.show()

# %matplotlib inline
# from pylab import rcParams
# rcParams['figure.figsize'] = 10, 5
# 
# cmap = plt.get_cmap('hot')
# colors = [cmap(i) for i in np.linspace(0, 1, n_steps)]
# for i in range(1,len(best_models)):
#     plt.plot(range(0,101), cgains_all[i], color=colors[i-1])
# plt.plot(range(0,101), cgains_all[-1], color="black")
#             
# axes = plt.gca()
# axes.set_ylim([0,max(max(l) for l in np.array(cgains_all)[1:])])
# plt.show()

# ---

# ### Variable Importance

# Define function

# In[32]:

# Compute variable importance based on correlation between predictor and prediction (on selection set)
def getImportance(model):
    from scipy import stats
    
    predictors = [pred[2:] for pred in model.predictor_names]
    pearcorr = []
    for predictor in predictors:
        pearsonr = stats.pearsonr(x_selection.loc[:,'D_'+predictor].values, model.pred_selection[:,0])
        pearcorr.append(pearsonr[0].round(2))
    df_result = pd.DataFrame({'variable':predictors,'importance':pearcorr}, columns=['variable','importance'])
    return df_result


# Execute function

# In[33]:

importance_all=[None]
for i in best_models.index:
    importance_all.append(getImportance(best_models.loc[i,:]))


# Inspect

# %matplotlib inline
# from pylab import rcParams
# rcParams['figure.figsize'] = 10, 5
# 
# #nvars = optimal_nvars
# nvars =  len(best_models)
# 
# fig, ax = plt.subplots()
# predictors = importance_all[nvars].variable
# y_pos = np.arange(len(predictors))
# importance = importance_all[nvars].importance
# 
# ax.barh(y_pos, importance, align='center',
#         color='darkblue', ecolor='black')
# ax.set_yticks(y_pos)
# ax.set_yticklabels(predictors)
# ax.invert_yaxis()
# ax.set_xlabel('Importance')
# plt.show()

# ---

# ### Model Coefficients

# In[35]:

# Store all variable names + coefficients for every best model (with 1,2,3,... variables) 
vars_out = []
coef_out = []
nmod_out = []
for i in best_models.index:
    modout = best_models.loc[i,:]
    vars_out_st = ['Intercept']+[var[2:] for var in modout.predictor_names]
    vars_out.append(vars_out_st)
    coef_out_st = list(modout.modelfit.intercept_)+list(+ modout.modelfit.coef_[0])
    coef_out.append(coef_out_st)
    nmod_out.append([i]*(i+1))
    
vars_out = list(chain.from_iterable(vars_out))
coef_out = list(chain.from_iterable(coef_out))
nmod_out = list(chain.from_iterable(nmod_out))


# In[36]:

df_coeff = pd.DataFrame({'nstep':nmod_out,'varname':vars_out,'coeff':coef_out}, columns=['nstep','varname','coeff'])


# ---

# ### Export Files

# In[37]:

nmods = len(best_models)


# Auc curve

# In[38]:

filename = root+"/data/modeling/"+modelrun+"_auccurve.csv"
with open(filename, 'w') as csvfile:
    write=csv.writer(csvfile, delimiter =';')
    write.writerow(["optimal" ,optimal_nvars])
    write.writerow(["selected",optimal_nvars])
    write.writerow(["variable","train", "selection","validation"])
    write.writerows([best_models.predictor_lastadd[i][2:]
                     , best_models.auc_train[i].round(3) 
                     , best_models.auc_selection[i].round(3)
                     , best_models.auc_validation[i].round(3) ] for i in range(1,nmods+1))


# Cresp

# In[39]:

for v in range(1,nmods+1):
    filename = root+"/data/modeling/"+modelrun+"_cresp_"+str(v)+".csv"
    with open(filename, 'w') as csvfile:
        write=csv.writer(csvfile, delimiter =';') 
        write.writerows([i+1, cresp_all[v][i]] for i in range(0,100))


# Cgains

# In[40]:

for v in range(1,nmods+1):
    filename = root+"/data/modeling/"+modelrun+"_cgains_"+str(v)+".csv"
    with open(filename, 'w') as csvfile:
        write=csv.writer(csvfile, delimiter =';') 
        write.writerows([i, cgains_all[v][i]] for i in range(0,101))


# Variable importance

# In[41]:

for v in range(1,nmods+1):
    filename = root+"/data/modeling/"+modelrun+"_importance_"+str(v)+".csv"
    with open(filename, 'w') as csvfile:
        write=csv.writer(csvfile, delimiter =';') 
        write.writerow(['variable','importance'])
        write.writerows([importance_all[v].iloc[i,0],importance_all[v].iloc[i,1]] for i in range(v))


# Model coefficients

# In[42]:

out_path = root+"/data/modeling/"+modelrun+"_modelcoeff.csv"
df_coeff.to_csv(path_or_buf=out_path, sep=';', index=False, encoding='utf-8', line_terminator='\n', quoting=csv.QUOTE_NONNUMERIC)


# Log messages

# In[43]:

log.append("-- Modeling phase completed --"+"\n")


# In[44]:

log_file = open(root+"/python/"+modelrun+"_modeling.log",'w')
log_file.write('\n'.join(log))
log_file.close()


# ---

# ### Stop script

# In[45]:

print("ok")


# ---

# # WIP

# Scoring all rows

# # Scoring of all rows
# import re
# tic = time.time()
# df_score = pd.DataFrame([])
# df_score['ID'] = df_in['ID']
# scores = []
# for i in range(len(df_in)):
#     ### METHOD 1: using function
#     score = [optifit.predict_proba(df_in[optivars])[i,:][-1]]
#     ### METHOD 2: with coefficients (same method as in scoring)
#     #exponent = optiint + ((df_in[optivars].iloc[i,:])*(opticoef[0])).sum()
#     #score = [(math.exp(exponent)) / (1+math.exp(exponent))]
#     
#     scores.extend(score)
#     try:
#         zeros = re.findall('[0]+$',str(i))
#         if len(zeros[0])>=3:
#             print(i)
#     except:
#         a=1
# df_score['score']=pd.Series(scores)
# tac = time.time()
# print((tac-tic)/60)
# 
# 
# df_in.to_csv('df_mod.csv', sep=';', index=False, encoding='utf-8', line_terminator='\n')
# df_score.to_csv(path_or_buf='scores_modeling.csv', sep=';', index=False, encoding='utf-8', line_terminator='\n')

# # /WIP
