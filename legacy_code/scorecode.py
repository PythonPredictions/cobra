### Importing libraries & basetable to score
# Importing Libraries
import time
import math
import csv
import re
import pandas as pd
import numpy as np
# Importing Types
typevariables=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'TARGET', 'ID', 'scont_1', 'scont_2', 'scont_3', 'scont_4', 'scont_5', 'scont_6', 'scont_7', 'scont_8', 'scont_9', 'scont_10', 'scat_1', 'scat_2', 'scat_3', 'scat_4', 'scat_5', 'sflag_1', 'sflag_2', 'sflag_3', 'sflag_4', 'sflag_5']
typetypes=['int', 'str', 'int', 'str', 'int', 'str', 'str', 'str', 'str', 'str', 'int', 'int', 'int', 'str', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'str', 'str', 'str', 'str', 'str', 'int', 'int', 'int', 'int', 'int']
df_types=pd.DataFrame({'var':typevariables,'type':typetypes},columns=['var','type'])
df_types_copy = df_types.copy()
bool_mask = df_types_copy.loc[:,'type']!='bool'
df_types_copy.loc[bool_mask,'type'] = [getattr(__builtins__, type_str) for type_str in df_types_copy.loc[bool_mask,'type']]
df_types_copy.loc[bool_mask==False,'type'] = getattr(__builtins__, 'str')
types = df_types_copy.set_index('var').T.to_dict('records')
# Importing Basetable with similar typing as in univariate analysis
df_base = pd.read_csv('df_base.csv',header=0,sep=None,engine='python',converters=types[0])

### Creating dataframe containing model rules
modvariables=['Intercept', 'relationship', 'education', 'capital-gain', 'occupation', 'hours-per-week', 'marital-status', 'workclass']
modcoefficients=[-7.089251945907992, 2.857566239388173, 3.9559266007760656, 3.805293440526549, 3.0368562075260153, 2.8679282172225298, 3.8409050507426072, 1.270430053023763]
df_modrules=pd.DataFrame({'varname':modvariables,'coeff':modcoefficients})

### Creating dataframe containing incidence translation rules
prepvariables=['relationship', 'relationship', 'relationship', 'relationship', 'relationship', 'relationship', 'education', 'education', 'education', 'education', 'education', 'education', 'education', 'education', 'education', 'education', 'education', 'education', 'education', 'education', 'education', 'capital-gain', 'capital-gain', 'occupation', 'occupation', 'occupation', 'occupation', 'occupation', 'occupation', 'occupation', 'occupation', 'occupation', 'occupation', 'occupation', 'occupation', 'hours-per-week', 'hours-per-week', 'hours-per-week', 'hours-per-week', 'marital-status', 'marital-status', 'marital-status', 'marital-status', 'marital-status', 'marital-status', 'marital-status', 'workclass', 'workclass', 'workclass', 'workclass', 'workclass', 'workclass', 'workclass']
prepbins=['Husband', 'Wife', 'Not-in-family', 'Own-child', 'Unmarried', 'Other-relative', 'Some-college', '7th-8th', 'HS-grad', 'Bachelors', '5th-6th', 'Doctorate', 'Non-significants', 'Masters', 'Prof-school', '10th', '11th', '12th', '9th', '1st-4th', 'Preschool', '[..., 2105]', '(2105,...]', 'Exec-managerial', 'Non-significants', 'Sales', '?', 'Prof-specialty', 'Farming-fishing', 'Handlers-cleaners', 'Other-service', 'Protective-serv', 'Machine-op-inspct', 'Adm-clerical', 'Priv-house-serv', '(48,...]', '(40, 48]', '(35, 40]', '[..., 35]', 'Married-civ-spouse', 'Divorced', 'Never-married', 'Married-spouse-absent', 'Separated', 'Non-significants', 'Widowed', 'Private', 'Self-emp-not-inc', '?', 'Local-gov', 'Self-emp-inc', 'Non-significants', 'Federal-gov']
prepincids=[0.4474003641513251, 0.4803571428571429, 0.10295278698878887, 0.015203145478374838, 0.056633663366336635, 0.04037685060565276, 0.18898367952522252, 0.0641025641025641, 0.16431451612903225, 0.42006504878659, 0.05303030303030303, 0.7353951890034365, 0.2429441062534588, 0.5435276305828918, 0.7382198952879581, 0.05128205128205128, 0.05584281282316442, 0.08433734939759037, 0.04702970297029703, 0.060869565217391314, 0.022727272727272728, 0.20500310476359446, 0.6513859275053305, 0.4710552431359577, 0.23353535353535354, 0.2739825581395349, 0.10049893086243764, 0.4519454605919521, 0.11946308724832215, 0.06824644549763033, 0.043478260869565216, 0.3169734151329243, 0.11971372804163954, 0.14123893805309734, 0.015503875968992246, 0.4302445038011095, 0.3498478922207736, 0.207983367983368, 0.08491107286288009, 0.4459652889604581, 0.10560344827586207, 0.044271796769022084, 0.09846153846153846, 0.07431551499348109, 0.4736842105263158, 0.07603092783505154, 0.2187813366365835, 0.2839756592292089, 0.1010028653295129, 0.2814526588845655, 0.554320987654321, 0.26436781609195403, 0.3900709219858156]
df_prep = pd.DataFrame({'var':prepvariables,'bin':prepbins,'incid':prepincids}, dtype=object)
df_prep.loc[:,'incid']=df_prep.loc[:,'incid'].astype('float64')

### Grouping basetable predictors along their types and trimming basetable accordingly
predictors = list(df_modrules.loc[df_modrules.varname!='Intercept','varname'].values)
not_predictors = [column for column in df_base.columns if column not in predictors]
mask_FloatOrInt = (df_types.type=='int')|(df_types.type=='float')
numeric_headers=[var for var in df_types.loc[mask_FloatOrInt,'var'].values if var in predictors]
object_headers=[var for var in df_types.loc[df_types.type=='str','var'].values if var in predictors]
bool_headers=[var for var in df_types.loc[df_types.type=='bool','var'].values if var in predictors]
df_base = df_base[predictors+['ID']]

### Preprocessing the basetable
# Strip quot function
def strip_quot(x_in):
    try:
        x_out = x_in.strip().strip('"').strip("'")
    except:
        x_out=x_in
    return x_out
# Lower/upper function
def lower_upper(x_in):
    if ((x_in.lower() == 'id')|(x_in.lower() == 'target')):
        x_out = x_in.upper()
    else:
        x_out = x_in.lower()
    return x_out
# maskmissing function in str/bool columns
def maskmissing(var):
    crit1 = var.isnull()
    modvar = pd.Series([str(value).strip() for value in var])
    crit2 = modvar==pd.Series(['']*len(var))
    return crit1 | crit2
# Apply preprocessing functions
df_base = df_base.rename(columns=strip_quot)
df_base = df_base.rename(columns=lower_upper)
df_base = df_base.applymap(strip_quot)
for header in object_headers+bool_headers:
    mask = maskmissing(df_base[header])
    df_base.loc[mask,header]='Missing'

### Incidence replacement
# Recipient dataframe
df_out = pd.DataFrame()
df_out['ID']=df_base['ID']
# Incidence replacement for string columns
for header in object_headers+bool_headers:
    mask = df_prep.loc[:,'var']==header
    bins = df_prep.loc[mask,'bin']
    incidences = df_prep.loc[mask,'incid']
    nonsig_bins = []
    nonsig_incidences = []
    if (bins == 'Non-significants').any():
        nonsig_bins = [binn for binn in df_base[header].unique() if binn not in list(bins)]
        nonsig_incidences = list(incidences[bins=='Non-significants'])*len(nonsig_bins)
    keys = list(bins)
    keys.extend(nonsig_bins)
    values = list(incidences)
    values.extend(nonsig_incidences)
    keys_and_values = zip(keys,values)
    transdic = dict(keys_and_values)
    items_to_translate = df_base[header] 
    df_out.loc[:,'D_'+header]= pd.Series([transdic[item] for item in items_to_translate])
# Incidence replacement for numeric columns
for header in numeric_headers:
    mask = df_prep.loc[:,'var']==header
    bins = df_prep.loc[mask,'bin']
    incidences = df_prep.loc[mask,'incid']
    index_missing = bins.index[bins=='Missing']
    incidence_missing = incidences[index_missing]
    upper_values = pd.Series([])
    for i,binn in enumerate(bins.values):
        upper_value = binn.split(',')[-1]
        try:
            upper_value = re.findall('[0-9]+',upper_value)[0]
        except:
            upper_value = math.inf
        upper_values[i] = upper_value
    upper_values.index = bins.index
    upper_values.drop(index_missing, inplace=True)
    upper_values = upper_values.astype(float)
    upper_values.sort_values(inplace=True)
    upper_values_incidences = incidences[upper_values.index]
    upper_values.reset_index(drop=True, inplace=True)
    upper_values_incidences.reset_index(drop=True, inplace=True)
    mask_npnan = df_base.loc[:,header].isnull()
    lowest_memberships = upper_values.searchsorted(df_base.loc[:,header],side='left')
    incidences_to_attribute = upper_values_incidences[lowest_memberships].reset_index(drop=True)
    incidences_to_attribute[mask_npnan] = incidence_missing
    df_out['D_'+header] = incidences_to_attribute

### Scoring
df_scores = pd.DataFrame([])
df_scores['ID'] = df_out['ID']
scores = []
intercept=-7.08925194591
coefficients=np.array([2.857566239388173, 3.9559266007760656, 3.805293440526549, 3.0368562075260153, 2.8679282172225298, 3.8409050507426072, 1.270430053023763])
productsums = (df_out['D_'+pd.Series(predictors)]*coefficients).sum(axis=1)
exponents = intercept + productsums
scores = exponents.apply(func=lambda x:(math.exp(x)) / (1+math.exp(x)))
df_scores['score']=scores

