#%%
import pandas as pd
import numpy as np
from random import shuffle
from scipy import stats
from typing import Dict, Tuple
import sys

sys.path.insert(0,"C:/Local/pers/Documents/GitHub/Cobra/dev/preprocessor")

import preprocessor.categorical_regrouper as pr

import logging
log = logging.getLogger(__name__)

ROOT = "C:/Local/pers/Documents/GitHub/Cobra/"
df_data = pd.read_csv(ROOT + "datasets/titanic_data.csv")
df_data.rename(columns={'Survived': 'TARGET'}, inplace=True)
df_data['Pclass'] = df_data['Pclass'].astype(object)

split = ['TRAIN']*int(df_data.shape[0]*0.5) + \
        ['TEST']*int(df_data.shape[0]*0.2)+ \
        ['VALIDATION']*int(np.ceil(df_data.shape[0]*0.3))

shuffle(split)

df_data['PARTITION'] = split

df_x = pd.DataFrame(df_data[['Pclass', 'Embarked']][df_data['PARTITION'] == "TRAIN"])
df_y = df_data['TARGET'][df_data['PARTITION'] == "TRAIN"]


#%%
""" NEW SOLUTION """
CR = pr.CategoryRegrouper()

CR.fit(X=df_x, y=df_y, columns=["Embarked", "Pclass"])
print(CR.all_category_map_)
df_X_tr = CR.transform(X=df_x, columns=["Embarked", "Pclass"])

#%%
""" OLD SOLUTION """







#%%
