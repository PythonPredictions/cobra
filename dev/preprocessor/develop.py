#%%
import pandas as pd
import numpy as np
from random import shuffle
from scipy import stats
from typing import Dict, Tuple

import logging
log = logging.getLogger(__name__)

ROOT = "C:/Local/pers/Documents/GitHub/Cobra/"
df_data = pd.read_csv(ROOT + "datasets/titanic_data.csv")
df_data.rename(columns={'Survived': 'TARGET'}, inplace=True)

split = ['TRAIN']*int(df_data.shape[0]*0.5) + \
        ['TEST']*int(df_data.shape[0]*0.2)+ \
        ['VALIDATION']*int(np.ceil(df_data.shape[0]*0.3))

shuffle(split)

df_data['PARTITION'] = split

df_x = pd.DataFrame(df_data[['Parch', 'Embarked']][df_data['PARTITION'] == "TRAIN"])
df_y = df_data['TARGET'][df_data['PARTITION'] == "TRAIN"]


#%%
column = 'Embarked'
scale_cont = True
incidence_mean = df_y.mean()
pval_thresh = 0.001
keep_categories = []
keep = 'Missing'
category_map = {}
replace_with = 'non-significant'

for category in grps:
    #category = 'S'
    df_aux = pd.concat([df_x[column], df_y], axis=1)
    df_aux['obs_other'] = np.where(df_aux[column] == category, 0, 1)
    
    cont_table = pd.crosstab(df_aux['obs_other'], df_aux['TARGET'], margins=False)

    if scale_cont:
        size_other_cat = cont_table.iloc[1].sum()
        cont_table.iloc[1, 0] = (1-incidence_mean)*size_other_cat
        cont_table.iloc[1, 1] = incidence_mean*size_other_cat
        cont_table = cont_table.values.astype(np.int64)

    pval = stats.chi2_contingency(cont_table, correction=False)[1]
    #0.17914169501249405

    if pval<=pval_thresh:
        keep_categories.append(category)

if keep not in keep_categories and keep in df_x[column].unique().tolist():
    keep_categories.append(keep)

for category in df_x[column].unique().tolist():
    if category in keep_categories:
        category_map[category] = category
    else:
        category_map[category] = replace_with
    




#%%

class CategoryRegrouper():

    """ 
    TOOD
    -test the keep_categories and give warning if category not in column
    -transform will be just df.replace() with a dict
    -I am keeping missings, but not in the original code, inspect
    -ask geert about the _removeCategories function

    -write the rest
    -combine with categorical_processor.py
    -add to init
    -test if same as the old code
    -unit tests

    Regroups categories in categorical variables if based on signicicance
    with target variable

    Attributes
    ----------
    scale_cont : bool, default=True
        whether contingency table should be scaled before chi-2
    pval_thresh : float, default=0.001
        significance threshold for regroupping
    regroup_rename : str, default='non-significant'
        new name of non-significant regroupped variables
    missing_rename : str, default='Missing'
        new name of missing categories
    keep_missing : bool, default=True
        whether missing category should be kept in the result
    forced_categories : Dict, default=None
        dictionary to force categories -
            for each colum dict of {col:[forced vars]}
    """

    def __init__(self, scale_cont: bool=True,
                 pval_thresh: float=0.001,
                 regroup_rename: str="non-significant",
                 missing_rename: str="Missing",
                 keep_missing: bool=True,
                 forced_categories: Dict=None):
        self.scale_cont = scale_cont
        self.pval_thresh = pval_thresh
        self.regroup_rename = regroup_rename
        self.missing_rename = missing_rename
        self.keep_missing = keep_missing
        self.forced_categories = forced_categories

    def fit(self):
        pass

    def _fit_column(self, X: pd.DataFrame,
                    y: pd.Series,
                    column: str) -> Dict:

        category_map = {}
        keep_categories = []
        self.incidence_mean = y.mean()
        all_uq_categories = X[column].unique().tolist()

        # Rename target
        y.rename("TARGET", inplace=True)

        # Replace missings
        X = self._replaceMissings(X=X, column=column,
                                  replace_with=self.missing_rename)

        # Remove small categories
        categories = self._removeCategories(X=X, y=y, column=column)

        # Inspect remaining categories and test significance
        for category in categories:
            df_aux = pd.concat([X[column], y], axis=1)
            df_aux['other_cats'] = np.where(df_aux[column] == category, 0, 1)
            cont_table = pd.crosstab(index=df_aux['other_cats'],
                                     columns=df_aux['TARGET'],
                                     margins=False)

            # if true, we scale the "other" categories
            if self.scale_cont:
                size_other_cats = cont_table.iloc[1].sum()
                cont_table.iloc[1, 0] = (1-self.incidence_mean)*size_other_cats
                cont_table.iloc[1, 1] = self.incidence_mean*size_other_cats
                cont_table = cont_table.values.astype(np.int64)

            pval = stats.chi2_contingency(cont_table, correction=False)[1]      

            # If significant, keep it
            if pval <= self.pval_thresh:
                keep_categories.append(category)

        # Keep "Missing" even if it wasn't selected if
        # it is in the original categories and set to True
        if ((self.missing_rename not in keep_categories) and
           (self.missing_rename in all_uq_categories) and self.keep_missing):
            keep_categories.append(self.missing_rename)

        # Keep forced categories
        if self.forced_categories is not None:
            # If doesnt exists, give warning
            forced = [col for col in self.forced_categories[column]
                      if col in all_uq_categories]

            # Extend list and remove duplicates
            keep_categories = list(set(keep_categories.extend(forced)))

            difference = set(forced) - set(self.forced_categories[column])
            if len(difference) > 0:
                log.warning("Following forced categories: {} "
                            "are not in column: {}.".format(difference,
                                                            column))

        # Return dictionary as {old column : new column}
        for category in all_uq_categories:
            if category in keep_categories:
                category_map[category] = category
            else:
                category_map[category] = self.regroup_rename

        return category_map

    def transform(self):
        pass

    def _transform_column(self):
        pass

    def fit_transform(self):
        pass

    def _replaceMissings(self, X: pd.DataFrame,
                         column: str,
                         replace_with: str='Missing') -> pd.DataFrame:
        """
        Method replaces missing and empty cells with `Missing` (default) in
        a pd.DataFrame

        df_tst = _replaceMissings(X=df_x, column='Embarked')

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe where a value will be replaced if empty or nan
        column : str
            Column to be analyzed for missings
        replace_with : str default='Missing'
            string to replace the missings

        Raises
        ------
        ValueError
            in case input column is not a string

        Returns
        -------
        pd.DataFrame
            modified dataframe with replaced missings
        """
        if not X[column].dtype == 'object':
            raise TypeError("columns must be a string")

        X[column].fillna(replace_with, inplace=True)
        X[column] = X[column].str.strip()
        X[column].replace('', replace_with, inplace=True)

        return X

    def _removeCategories(self, X: pd.DataFrame,
                          y: pd.Series,
                          column: str,
                          threshold: int=5) -> np.ndarray:
        """
        Method removes category which fail to meet certain condition

        grps = _removeGroups(X=df_x, y=df_y, column='Embarked')

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe with columns to be inspected for group removal
        y : pd.Series
            Series with target
        column : str
            Column to be analyzed group removal
        threshold : int default=5
            Threshold for group removal

        Returns
        -------
        np.ndarray
            numpy array with groups to be kept
        """
        category_cnts = pd.DataFrame(X.groupby(column)[column].count())
        train_inc = y.mean()
        factor = max(train_inc, 1-train_inc)
        keep_categories = category_cnts.where((category_cnts*factor) >
                                              threshold)

        return np.array(keep_categories.index.tolist())



#%%
CR = CategoryRegrouper()

output = CR._fit_column(X=df_x, y=df_y, column='Embarked')
output

#%%
