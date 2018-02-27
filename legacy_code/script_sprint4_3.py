import pandas as pd
import os

try:
    root = os.path.dirname(os.path.realpath(__file__))
    root = "/".join(root.split('\\')[:-1])
except:
    root = 'C:\/wamp64\/www\/python_predictions_4\/assets\/scripts\/python'

auc_path = root + '\/data\/univariate\/aucs.csv'
df_in = pd.read_csv(auc_path, sep=';')
df_sortqual = df_in.sort_values(by=['AUC test','AUC train'], ascending=False).reset_index(drop=True)
df_sortname = df_in.sort_values(by=['variable']).reset_index(drop=True).reset_index(drop=True)
if (df_in.variable == df_sortqual.variable).all():
    df_out = df_sortname
else:
    df_out = df_sortqual

df_out.to_csv(path_or_buf=auc_path
              ,sep=';'
              ,index=False
              ,encoding='utf-8'
              ,line_terminator='\n')
			  
print('ok')
