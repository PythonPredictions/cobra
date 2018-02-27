
# coding: utf-8

# # Scoring

# Purpose of this script is to output (an)other script(s) with self-contained code to score out a basetable
# 
# Call this script: the scriptmaker
# 
# Call the created self-contained script: the scoring script

# ### Import libraries

# In[1]:

import time
import math
import csv
import re
import pandas as pd
import numpy as np


# ---

# ### Miscellaneous

# In[2]:

log = []


# In[3]:

# When code is in script, we define the path of the script's parent folder location as the root directory
# From this root we can travel to the relevant folders with minimal adjustment
try:
    root = os.path.dirname(os.path.realpath(__file__))
    root = "/".join(root.split('\\')[:-1])
    log.append('Dynamic paths'+'\n')
except:
    root = 'C:/wamp64/www/python_predictions_4/assets/scripts'
    log.append('Static paths'+'\n')


# ---

# ### Read the data and create variables to be exported

# ##### 1A - Retrieve Modeltab info to find out for which modeltab we want a scoring script for
# Intermediate step for 1C

# In[4]:

df_modeltab = pd.read_csv(root+'/data/univariate/modeltab_info.csv',sep=';', index_col=0, header=None).T
modeltabtoscore = df_modeltab.score[1]


# ##### 1B -  Retrieve number of vars selected to identify which n-th model of the 'modeltab'-models we are interested in
# Intermediate step for 1C

# In[5]:

df_auccurve_path = root+"/data/modeling/"+modeltabtoscore+"_auccurve.csv"
df_auccurve = open(df_auccurve_path).read()
selected_nvars = int(re.findall(r"selected;[0-9]+",df_auccurve)[0].split(';')[-1])
selected_nvars = len(pd.read_csv(df_auccurve_path,skiprows=3, sep=';'))  # USED FOR TESTING ALL VARS, TO BE DELETED !!!!!!!!!!!


# ##### 1C - Retrieve model coefficients of the specific n-th model of the 'modeltab'-models
# 
# Result is to be stored in text in the scoring script, to achieve self-containment

# In[6]:

df_modelcoeff_path = root+"/data/modeling/"+modeltabtoscore+"_modelcoeff.csv"
df_modelcoeff = pd.read_csv(df_modelcoeff_path, sep=';')
mask = df_modelcoeff.nstep == int(selected_nvars)
df_modrules = df_modelcoeff.loc[mask,:] # TO BE STORED in SCORING SCRIPT -------------------------------------------------------


# ##### 2 - Retrieve Data types of the predictors
# 
# Result is to be stored in text in the scoring script, to achieve self-containment

# In[7]:

types_path = root+"/python/data_types.csv"
df_types = pd.read_csv(types_path, header=None)
df_types.columns=['var','type'] # TO BE STORED in SCORING SCRIPT ---------------------------------------------------------------


# ##### 3 - Univariate table output for deriving the translation from original VAR to discretisized D_VAR
# Result is to be stored in text in the scoring script, to achieve self-containment

# In[8]:

df_univariate_path = root+"/data/univariate/df_univariate.csv"
# we create str converters for all the B_variables coming from the univariate table
# we need this because bool & str variables (defined by ...types.csv) from the basetable are converted to objects
#  and will have to be compared to the B_variables values, which better have the same type
#  e.g. we have a varflag in our basetable which is converted to an object, but assume B_varflag is not converted and will be automatically read as float 1.0/0.0
#       in our incidence replacement we will thus be comparing '1'/'0' with 1.0/0.0, which won't work 
uni_iterable = [(variable,getattr(__builtins__, 'str')) for variable in  'B_'+df_modrules.varname[1:].values]
uni_dict = dict(uni_iterable)
df_uni = pd.read_csv(df_univariate_path, sep=";", converters=uni_dict)


# In[9]:

dcolumns = ['D_'+name for name in df_modrules.varname[1:]]
bcolumns = ['B_'+name for name in df_modrules.varname[1:]]
gvar = []
gincid = []
gbin = []

for i in range(len(dcolumns)):
    # Select B_varname and D_varname
    # Then take unique combinations of B_var and D_var in the univariate dataframe
    # These combinations give the incidence value to attribute to the (possibly discretisized/regrouped) variables
    columns_set = dcolumns[i:i+1]+bcolumns[i:i+1]
    df_dupli = df_uni.loc[:,columns_set].drop_duplicates()
    n_occurences = len(df_dupli) 
    
    gvar.extend([df_dupli.columns[0][2:]]*n_occurences)
    gincid.extend(df_dupli.iloc[:,0].values)
    gbin.extend(df_dupli.iloc[:,1].values)

    
df_prep = pd.DataFrame({'var':gvar,'bin':gbin,'incid':gincid} 
                       ,columns=['var','bin','incid']) # TO BE STORED in SCORING SCRIPT --------------------------------


# ---

# ### Writing scoring scripts

# ##### for R

# In[10]:

score_code = open(root+"/Python/scorecode.R",'w')

score_code.write("### Importing libraries & basetable to score\n")
score_code.write("# Importing libraries\n")
score_code.write("#library(dplyr)\n")
score_code.write("# Importing Types\n")
score_code.write("typevariables=c"+str([var for var in df_types.loc[:,'var']]).replace("[","(").replace("]",")")+"\n")
score_code.write("typetypes=c"+str([vartype for vartype in df_types.loc[:,'type']]).replace("[","(").replace("]",")")+"\n")
score_code.write("df_types=data.frame(var=typevariables,type=typetypes, stringsAsFactors='False')\n")
score_code.write("df_types_copy = df_types\n")
score_code.write("df_types_copy$type[df_types_copy$type=='int'|df_types_copy$type=='float']='numeric'\n")
score_code.write("df_types_copy$type[df_types_copy$type=='str'|df_types_copy$type=='bool']='character'\n")
score_code.write("coltypes = df_types_copy$type\n")
score_code.write("names(coltypes) = df_types_copy$var\n")
score_code.write("# Importing Basetable (with similar typing as in univariate analysis)\n")
score_code.write("df_base = read.csv('df_base.csv', check.names='False', colClasses=coltypes )\n")

score_code.write("### Creating dataframe containing model rules\n")
score_code.write("modvariables=c"+str([var for var in df_modrules.loc[:,'varname']]).replace("[","(").replace("]",")")+"\n")
score_code.write("modcoefficients=c"+str([coeff for coeff in df_modrules.loc[:,'coeff']]).replace("[","(").replace("]",")")+"\n")
score_code.write("df_modrules=data.frame(varname=modvariables,coeff=modcoefficients, stringsAsFactors='False')\n")
score_code.write("\n")

score_code.write("### Creating dataframe containing incidence translation rules\n")
score_code.write("prepvariables=c"+str([var for var in df_prep.loc[:,'var']]).replace("[","(").replace("]",")")+"\n")
score_code.write("prepbins=c"+str([bin for bin in df_prep.loc[:,'bin']]).replace("[","(").replace("]",")")+"\n")
score_code.write("prepincids=c"+str([bin for bin in df_prep.loc[:,'incid']]).replace("[","(").replace("]",")")+"\n")
score_code.write("df_prep =data.frame(var=prepvariables,bin=prepbins,incid=prepincids, stringsAsFactors='False')\n")
score_code.write("\n")

score_code.write("### Grouping basetable predictors along their types and trimming basetable accordingly\n")
score_code.write("predictors = df_modrules$varname[df_modrules$varname!='Intercept']\n")
score_code.write("not_predictors = subset(colnames(df_base),!(colnames(df_base) %in% predictors))\n")
score_code.write("mask_FloatOrInt = df_types$type=='int'|df_types$type=='float'\n")
score_code.write("numeric_headers = subset(df_types$var[mask_FloatOrInt], df_types$var[mask_FloatOrInt] %in% predictors)\n")
score_code.write("object_headers = subset(df_types$var[df_types$type=='str'], df_types$var[df_types$type=='str'] %in% predictors)\n")
score_code.write("bool_headers = subset(df_types$var[df_types$type=='bool'], df_types$var[df_types$type=='bool'] %in% predictors)\n")
score_code.write("df_base = df_base[c(predictors,'ID')]\n")
score_code.write("\n")

score_code.write("### Preprocessing the basetable\n")
score_code.write("# Strip quot function\n")
score_code.write("strip_quot<-function(x){\n")
score_code.write('    x = gsub("')
score_code.write("'")
score_code.write('","",x)\n')
score_code.write("    x = gsub('")
score_code.write('"')
score_code.write("','',x)\n")               
score_code.write("    x = trimws(x)\n")
score_code.write("    return(x)\n")
score_code.write("}\n")
score_code.write("# Lower/upper function\n")
score_code.write("lower_upper<-function(x){\n")
score_code.write("    if (tolower(x)=='id'|tolower(x)=='target'){\n")
score_code.write("        x = toupper(x)\n")
score_code.write("    }\n")
score_code.write("    else {\n")
score_code.write("        x = tolower(x)\n")
score_code.write("    }\n")
score_code.write("}\n")
score_code.write("# maskmissing function in str/bool columns\n")
score_code.write("maskmissing<-function(var){\n")
score_code.write("    crit1 = is.na(var)\n")
score_code.write("    crit2 = var==''\n")
score_code.write("    return(crit1|crit2)\n")
score_code.write("}\n")
score_code.write("# Apply preprocessing functions\n")
score_code.write("colnames(df_base) = sapply(colnames(df_base), lower_upper)\n")
score_code.write("colnames(df_base) = sapply(colnames(df_base), strip_quot)\n")
score_code.write("df_base[] = lapply(df_base, strip_quot)\n")
score_code.write("for (predictor in c(object_headers,bool_headers)){\n")
score_code.write("    df_base[maskmissing(df_base[predictor]),predictor]='Missing'\n")
score_code.write("}\n")
score_code.write("\n")

score_code.write("### Incidence replacement\n")
score_code.write("# Recipient dataframe\n")
score_code.write("df_out = data.frame(ID=df_base$ID)\n")
score_code.write("# Incidence replacement for string columns\n")
score_code.write("for (header in c(object_headers,bool_headers)){\n")
score_code.write("    mask = df_prep$var==header\n")
score_code.write("    bins = df_prep[mask,'bin']\n")
score_code.write("    incidences = df_prep[mask,'incid']\n")
score_code.write("    nonsig_bins = c()\n")
score_code.write("    nonsig_incidences = c()\n")
score_code.write("    if (sum(bins == 'Non-significants')>0) {\n")
score_code.write("        nonsig_bins = subset(unique(df_base[,header]), !(unique(df_base[,header]) %in% bins))\n")
score_code.write("        nonsig_incidences = rep(incidences[bins=='Non-significants'],length(nonsig_bins))\n")
score_code.write("    }\n")
score_code.write("    keys = c(bins,nonsig_bins)\n")
score_code.write("    values = c(incidences,nonsig_incidences)\n")
score_code.write("    df_out[paste('D_',header, sep='')] = values[match(df_base[,header], keys)]\n")
score_code.write("}\n")
score_code.write("# Incidence replacement for numeric columns\n")
score_code.write("for (header in numeric_headers){\n")
score_code.write("    mask = df_prep$var==header\n")
score_code.write("    bins = df_prep[mask,'bin']\n")
score_code.write("    incidences = df_prep[mask,'incid']\n")
score_code.write("    index_missing = which(bins=='Missing')\n")
score_code.write("    incidence_missing = incidences[index_missing]\n")
score_code.write("    upper_values = c()\n")
score_code.write("    last <- function(x) { return( x[length(x)] ) }\n")
score_code.write("    for (binn in bins){\n")
score_code.write("        upper_value = last(unlist(strsplit(binn,',')))\n")
score_code.write("        upper_value = tryCatch(as.numeric(gsub('([0-9]+).*$', '\\")
score_code.write("\\")
score_code.write("1',upper_value)), warning=function(e) Inf)\n")
score_code.write("        upper_values = c(upper_values,upper_value)\n")
score_code.write("    }\n")
score_code.write("    if(!identical(index_missing,integer(0))) upper_values = upper_values[-index_missing]\n")
score_code.write("    if(!identical(index_missing,integer(0))) incidences = incidences[-index_missing]\n")
score_code.write("    upper_values_incidences = incidences[order(upper_values)]\n")
score_code.write("    upper_values = upper_values[order(upper_values)]\n")
#score_code.write("    incidence_replaced_values = c()\n")
#score_code.write("    for (original_value in as.numeric(df_base[,header])){\n")
#score_code.write("        if (is.na(original_value)){\n")
#score_code.write("            incidence_to_attribute = incidence_missing\n")
#score_code.write("        }\n")
#score_code.write("        else {\n")
#score_code.write("            lowest_membership = min(which(original_value<=upper_values))\n")
#score_code.write("            incidence_to_attribute = upper_values_incidences[lowest_membership]\n")
#score_code.write("        }\n")
#score_code.write("        incidence_replaced_values = c(incidence_replaced_values,incidence_to_attribute)\n")
#score_code.write("    }\n")
#score_code.write("    df_out[paste('D_',header, sep='')] = incidence_replaced_values\n")
score_code.write("    mask_nan = is.na(df_base[,header])\n")
score_code.write("    lowest_memberships = findInterval(as.numeric(df_base[,header]), upper_values * (1 + .Machine$double.eps)) + 1\n")
score_code.write("    incidences_to_attribute = upper_values_incidences[lowest_memberships]\n")
score_code.write("    incidences_to_attribute[mask_nan] = incidence_missing\n")
score_code.write("    df_out[paste('D_',header, sep='')] = incidences_to_attribute\n")
score_code.write("}\n")
score_code.write("\n")

score_code.write("### Scoring\n")
score_code.write("df_scores = data.frame(ID=as.numeric(as.character(df_out$ID)))\n")
score_code.write("scores = c()\n")
score_code.write("intercept="+str(df_modrules.coeff.values[0])+"\n")
score_code.write("coefficients=c"+str([coeff for coeff in df_modrules.coeff][1:]).replace("[","(").replace("]",")")+"\n")
score_code.write("productsums = rowSums(t(t(df_out[,paste('D_',predictors,sep='')])*coefficients))\n")
score_code.write("exponents = intercept + productsums\n")
score_code.write("scores = sapply(exponents, FUN = function(x) (exp(x)) / (1+exp(x)))\n")
score_code.write("df_scores['score']=scores\n")
score_code.write("\n")

score_code.close()


# ##### for Python

# In[11]:

score_code = open(root+"/Python/scorecode.py",'w')

score_code.write("### Importing libraries & basetable to score\n")
score_code.write("# Importing Libraries\n")
score_code.write("import time\nimport math\nimport csv\nimport re\nimport pandas as pd\nimport numpy as np\n")
score_code.write("# Importing Types\n")
score_code.write("typevariables="+str([var for var in df_types.loc[:,'var']])+"\n")
score_code.write("typetypes="+str([vartype for vartype in df_types.loc[:,'type']])+"\n")
score_code.write("df_types=pd.DataFrame({'var':typevariables,'type':typetypes},columns=['var','type'])\n")
score_code.write("df_types_copy = df_types.copy()\n")
score_code.write("bool_mask = df_types_copy.loc[:,'type']!='bool'\n")
score_code.write("df_types_copy.loc[bool_mask,'type'] = [getattr(__builtins__, type_str) for type_str in df_types_copy.loc[bool_mask,'type']]\n")
score_code.write("df_types_copy.loc[bool_mask==False,'type'] = getattr(__builtins__, 'str')\n")
score_code.write("types = df_types_copy.set_index('var').T.to_dict('records')\n")                 
score_code.write("# Importing Basetable with similar typing as in univariate analysis\n")
score_code.write("df_base = pd.read_csv('df_base.csv',header=0,sep=None,engine='python',converters=types[0])\n")
score_code.write("\n")

score_code.write("### Creating dataframe containing model rules\n")
score_code.write("modvariables="+str([var for var in df_modrules.loc[:,'varname']])+"\n")
score_code.write("modcoefficients="+str([coeff for coeff in df_modrules.loc[:,'coeff']])+"\n")
score_code.write("df_modrules=pd.DataFrame({'varname':modvariables,'coeff':modcoefficients})\n")
score_code.write("\n")

score_code.write("### Creating dataframe containing incidence translation rules\n")
score_code.write("prepvariables="+str([var for var in df_prep.loc[:,'var']])+"\n")
score_code.write("prepbins="+str([bin for bin in df_prep.loc[:,'bin']])+"\n")
score_code.write("prepincids="+str([bin for bin in df_prep.loc[:,'incid']])+"\n")
score_code.write("df_prep = pd.DataFrame({'var':prepvariables,'bin':prepbins,'incid':prepincids}, dtype=object)\n")
score_code.write("df_prep.loc[:,'incid']=df_prep.loc[:,'incid'].astype('float64')\n")
score_code.write("\n")

score_code.write("### Grouping basetable predictors along their types and trimming basetable accordingly\n")
score_code.write("predictors = list(df_modrules.loc[df_modrules.varname!='Intercept','varname'].values)\n")
score_code.write("not_predictors = [column for column in df_base.columns if column not in predictors]\n")
score_code.write("mask_FloatOrInt = (df_types.type=='int')|(df_types.type=='float')\n")
score_code.write("numeric_headers=[var for var in df_types.loc[mask_FloatOrInt,'var'].values if var in predictors]\n")
score_code.write("object_headers=[var for var in df_types.loc[df_types.type=='str','var'].values if var in predictors]\n")
score_code.write("bool_headers=[var for var in df_types.loc[df_types.type=='bool','var'].values if var in predictors]\n")
score_code.write("df_base = df_base[predictors+['ID']]\n")
score_code.write("\n")

score_code.write("### Preprocessing the basetable\n")
score_code.write("# Strip quot function\n")
score_code.write("def strip_quot(x_in):\n")
score_code.write("    try:\n")
score_code.write("        x_out = x_in.strip().strip('")
score_code.write('"')
score_code.write("').strip(")
score_code.write('"')
score_code.write("'")
score_code.write('")\n')
score_code.write("    except:\n")
score_code.write("        x_out=x_in\n")
score_code.write("    return x_out\n")
score_code.write("# Lower/upper function\n")
score_code.write("def lower_upper(x_in):\n")
score_code.write("    if ((x_in.lower() == 'id')|(x_in.lower() == 'target')):\n")
score_code.write("        x_out = x_in.upper()\n")
score_code.write("    else:\n")
score_code.write("        x_out = x_in.lower()\n")
score_code.write("    return x_out\n")
score_code.write("# maskmissing function in str/bool columns\n")
score_code.write("def maskmissing(var):\n")
score_code.write("    crit1 = var.isnull()\n")
score_code.write("    modvar = pd.Series([str(value).strip() for value in var])\n")
score_code.write("    crit2 = modvar==pd.Series(['']*len(var))\n")
score_code.write("    return crit1 | crit2\n")
score_code.write("# Apply preprocessing functions\n")
score_code.write("df_base = df_base.rename(columns=strip_quot)\n")
score_code.write("df_base = df_base.rename(columns=lower_upper)\n")
score_code.write("df_base = df_base.applymap(strip_quot)\n")
score_code.write("for header in object_headers+bool_headers:\n")
score_code.write("    mask = maskmissing(df_base[header])\n")
score_code.write("    df_base.loc[mask,header]='Missing'\n")
score_code.write("\n")
                 
score_code.write("### Incidence replacement\n")
score_code.write("# Recipient dataframe\n")
score_code.write("df_out = pd.DataFrame()\n")
score_code.write("df_out['ID']=df_base['ID']\n")
score_code.write("# Incidence replacement for string columns\n")
score_code.write("for header in object_headers+bool_headers:\n")
score_code.write("    mask = df_prep.loc[:,'var']==header\n")
score_code.write("    bins = df_prep.loc[mask,'bin']\n")
score_code.write("    incidences = df_prep.loc[mask,'incid']\n")
score_code.write("    nonsig_bins = []\n")
score_code.write("    nonsig_incidences = []\n")
score_code.write("    if (bins == 'Non-significants').any():\n")
score_code.write("        nonsig_bins = [binn for binn in df_base[header].unique() if binn not in list(bins)]\n")
score_code.write("        nonsig_incidences = list(incidences[bins=='Non-significants'])*len(nonsig_bins)\n")
score_code.write("    keys = list(bins)\n")
score_code.write("    keys.extend(nonsig_bins)\n")
score_code.write("    values = list(incidences)\n")
score_code.write("    values.extend(nonsig_incidences)\n")
score_code.write("    keys_and_values = zip(keys,values)\n")
score_code.write("    transdic = dict(keys_and_values)\n")
score_code.write("    items_to_translate = df_base[header] \n")
score_code.write("    df_out.loc[:,'D_'+header]= pd.Series([transdic[item] for item in items_to_translate])\n")
score_code.write("# Incidence replacement for numeric columns\n")
score_code.write("for header in numeric_headers:\n")
score_code.write("    mask = df_prep.loc[:,'var']==header\n")
score_code.write("    bins = df_prep.loc[mask,'bin']\n")
score_code.write("    incidences = df_prep.loc[mask,'incid']\n")
score_code.write("    index_missing = bins.index[bins=='Missing']\n")
score_code.write("    incidence_missing = incidences[index_missing]\n")
score_code.write("    upper_values = pd.Series([])\n")
score_code.write("    for i,binn in enumerate(bins.values):\n")
score_code.write("        upper_value = binn.split(',')[-1]\n")
score_code.write("        try:\n")
score_code.write("            upper_value = re.findall('[0-9]+',upper_value)[0]\n")
score_code.write("        except:\n")
score_code.write("            upper_value = math.inf\n")
score_code.write("        upper_values[i] = upper_value\n")
score_code.write("    upper_values.index = bins.index\n")
score_code.write("    upper_values.drop(index_missing, inplace=True)\n")
score_code.write("    upper_values = upper_values.astype(float)\n")
score_code.write("    upper_values.sort_values(inplace=True)\n")
score_code.write("    upper_values_incidences = incidences[upper_values.index]\n")
score_code.write("    upper_values.reset_index(drop=True, inplace=True)\n")
score_code.write("    upper_values_incidences.reset_index(drop=True, inplace=True)\n")
#score_code.write("    incidence_replaced_values = np.array([])\n")
#score_code.write("    for original_value in df_base[header]:\n")
#score_code.write("        lowest_membership = upper_values.index[original_value<=upper_values].min()\n")
#score_code.write("        try:\n")
#score_code.write("            incidence_to_attribute = upper_values_incidences[lowest_membership]\n")
#score_code.write("        except:\n")
#score_code.write("            if np.isnan(original_value):\n")
#score_code.write("                incidence_to_attribute = incidence_missing\n")
#score_code.write("            else:\n")
#score_code.write("                incidence_to_attribute = np.nan\n")
#score_code.write("        incidence_replaced_values = np.append(incidence_replaced_values,incidence_to_attribute)\n")
#score_code.write("    df_out['D_'+header] = pd.Series(incidence_replaced_values)\n")
score_code.write("    mask_npnan = df_base.loc[:,header].isnull()\n")
score_code.write("    lowest_memberships = upper_values.searchsorted(df_base.loc[:,header],side='left')\n")
score_code.write("    incidences_to_attribute = upper_values_incidences[lowest_memberships].reset_index(drop=True)\n")
score_code.write("    incidences_to_attribute[mask_npnan] = incidence_missing\n")
score_code.write("    df_out['D_'+header] = incidences_to_attribute\n")
score_code.write("\n")    

                 
score_code.write("### Scoring\n")
score_code.write("df_scores = pd.DataFrame([])\n")
score_code.write("df_scores['ID'] = df_out['ID']\n")
score_code.write("scores = []\n")
score_code.write("intercept="+str(df_modrules.coeff.values[0])+"\n")
score_code.write("coefficients=np.array("+str([coeff for coeff in df_modrules.coeff][1:])+")\n")
score_code.write("productsums = (df_out['D_'+pd.Series(predictors)]*coefficients).sum(axis=1)\n")
score_code.write("exponents = intercept + productsums\n")
score_code.write("scores = exponents.apply(func=lambda x:(math.exp(x)) / (1+math.exp(x)))\n")
score_code.write("df_scores['score']=scores\n")
score_code.write("\n")

score_code.close()


# ##### for Sas

# In[12]:

# ...

print('ok')

# ---
