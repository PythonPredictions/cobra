#%%
import pandas as pd
import numpy as np
import sys
sys.path.insert(0,"C:/Local/pers/Documents/GitHub/Cobra")

ROOT = "C:/Local/pers/Documents/GitHub/Cobra/"

#%%
df_data = pd.read_csv(ROOT + "datasets/titanic_data.csv")

#%%
from cobra.preprocessing import KBinsDiscretizer

KBD = KBinsDiscretizer()
df_prep = KBD.fit_transform(data=df_data, column_names=['Age','Fare'])

#%%
from cobra.preprocessing import TargetEncoder