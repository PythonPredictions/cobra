import pandas as pd
import numpy as np




filename_modeltab = "C:\/wamp64\/www\/python_predictions_4\/assets\/scripts\/data\/univariate\/modeltab_info.csv"
filename_varsel = "C:\/wamp64\www\/python_predictions_4\/assets\/scripts\/data\/univariate\/variable_selections.csv"

# Variable_selections.csv is the file that needs to be modified 
df_varsel = pd.read_csv(filename_varsel, sep=";")

# From modeltab_info.csv we get the information which column in variable_selections.csv needs to be modified 
# as well as which model-tab_auccurve.csv needs to be imported to use as template-filler
df_tab = pd.read_csv(filename_modeltab, sep=";")
new = df_tab.loc[df_tab.key=="new", "value"].reset_index(drop=True)[0]
new_template = df_tab.loc[df_tab.key=="new_template", "value"].reset_index(drop=True)[0]

# The correct model-tab_auccurve.csv is imported from which the variables to be forced are read 
# This list can be deduced from the line "selected" and the column "variable"
template_list = []
if new_template.upper()!="SCRATCH":
	filename_auccurve = "C:\/wamp64\/www\/python_predictions_4\/assets\/scripts\/data\/modeling\/"+new_template+"_auccurve.csv"
	df_usersel = pd.read_csv(filename_auccurve, sep=";", nrows =2, header=None, names=["option","value"])
	nvars = df_usersel.loc[df_usersel.option == "selected", "value"].reset_index(drop=True)[0]
	df_template = pd.read_csv(filename_auccurve, sep=";", skiprows=2)
	template_list = list(df_template.loc[:nvars-1, "variable"])
mask_force= np.array([var in template_list for var in df_varsel.variable])


# Apply template to variable_selections.csv
df_varsel.loc[:,[new]] = 0
df_varsel.loc[mask_force,[new]] = 1

# Export to csv
df_varsel.to_csv(filename_varsel, sep=";", index=False)

print("ok")
