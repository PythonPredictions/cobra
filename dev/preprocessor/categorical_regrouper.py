
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
import logging

log = logging.getLogger(__name__)


class CategoryRegrouper(BaseEstimator, TransformerMixin):
    """
    Regroups categories in categorical variables based on significance
    with target variable.

    Parameters
    ----------
    scale_cont : bool, default=True
        Whether contingency table should be scaled before chi^2.'

    pval_thresh : float, default=0.001
        Significance threshold for regroupping.

    regroup_rename : str, default='non-significant'
        New name of non-significant regroupped variables.

    missing_rename : str, default='Missing'
        New name of missing categories.

    keep_missing : bool, default=Falsse
        Whether missing category should be kept in the result.

    forced_categories : Dict, default=None
        Dictionary to force categories -
            for each colum dict of {col:[forced vars]}.

    Attributes
    ----------
    all_category_map_ : Dict
        Dictionary with mapping for each variable.
    """
    def __init__(self, scale_cont: bool = True,
                 pval_thresh: float = 0.001,
                 regroup_rename: str = "non-significant",
                 missing_rename: str = "Missing",
                 keep_missing: bool = False,
                 forced_categories: Dict = None):
        self.scale_cont = scale_cont
        self.pval_thresh = pval_thresh
        self.regroup_rename = regroup_rename
        self.missing_rename = missing_rename
        self.keep_missing = keep_missing
        self.forced_categories = forced_categories

    def fit(self, X: pd.DataFrame,
            y: pd.Series,
            columns: list = []):
        """
        Method regroups categories whole DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe with all the columns.

        y: pd.Series
            Series with target variable.

        columns : list, default=[]
            Columns to be regrouped.

        Raises
        ------
        ValueError
            In case X and y are not of the same length.

        Returns
        -------
        None
            Only fits the instance of the class.
        """
        self.all_category_map_ = {}

        if len(X.index) != len(y.index):
            raise ValueError("The length of X is {}, but the length of y is {}"
                             .format(len(X.index), len(y.index)))

        if not columns:
            columns = CategoryRegrouper._get_categorical_columns(X)
            log.warning("All object-type columns have been selected")

        for column in columns:
            if column not in X.columns:
                log.warning("DataFrame has no column '{}', so it will be "
                            "skipped in fitting" .format(column))
                continue

            self.all_category_map_[column] = self._fit_column(X=X,
                                                              y=y,
                                                              column=column)

    def _fit_column(self, X: pd.DataFrame,
                    y: pd.Series,
                    column: str) -> Dict:
        """
        Method regroups categories in given column.

        Parameters
        ----------
        X : pd.Series
            Series with one column to be transformed.

        y: pd.Series
            Series with target variable

        column : str
            Column to be regrouped.

        Raises
        ------
        ValueError
            in case input column is not a string.

        Returns
        -------
        Dict
            Returns dictionary as {old category : new category} for
            specific column.
        """
        category_map = {}
        keep_categories = []
        incidence_mean = y.mean()

        # Rename target
        y.rename("TARGET", inplace=True)

        # Replace missings
        X = self._replaceMissings(X=X, column=column,
                                  replace_with=self.missing_rename)

        all_uq_categories = X[column].unique().tolist()

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
                cont_table.iloc[1, 0] = (1-incidence_mean)*size_other_cats
                cont_table.iloc[1, 1] = incidence_mean*size_other_cats
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

    def transform(self, X: pd.DataFrame,
                  columns: list = []) -> pd.DataFrame:
        """
        Method transforms specified columns.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe with all the columns.

        columns : list, default=[]
            Columns to be regrouped.

        Raises
        ------
        NotFittedError
            If fit() method has not been called.

        ValueError
            If columns to be transformed have not been fitted.

        Returns
        -------
        pd.DataFrame
            Returns transformed DataFrame with new columns as "col_regrouped".
        """
        if len(self.all_category_map_) == 0:
            msg = ("This {} instance is not fitted yet. Call 'fit' with "
                   "appropriate arguments before using this method.")

            raise NotFittedError(msg.format(self.__class__.__name__))

        fitted_columns = list(self.all_category_map_.keys())

        # if specified columns not in fitted Dict, raise error
        if not set(columns).issubset(set(fitted_columns)):
            diff_cols = set(columns).difference(set(fitted_columns))
            raise ValueError("Following columns are not fitted: "
                             "{}".format(diff_cols))

        X_tr = X.copy()
        for column in columns:
            X_tr[column + "_regrouped"] = self._transform_column(X=X,
                                                                 column=column)

        return X_tr

    def _transform_column(self, X: pd.DataFrame,
                          column: str) -> pd.Series:
        """
        Method transforms specified columns.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe with all the columns.

        column : str
            Column to be regrouped.

        Returns
        -------
        pd.Series
            Returns DataFrame with regrouped variable as category datatype.
        """
        X_tr = X[column].copy()
        X_tr[column + "_regrouped"] = X_tr.replace(
                            to_replace=self.all_category_map_[column])

        X_tr[column + "_regrouped"] = X_tr[column +
                                           "_regrouped"].astype('category')

        return X_tr[column + "_regrouped"]

    def fit_transform(self, X: pd.DataFrame,
                      y: pd.Series,
                      columns: list = []) -> pd.DataFrame:
        """
        Auxiliary method fits and transforms specified columns.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe with all the columns.

        y : pd.Series
            Series with target variable

        column : list, default=[]
            Columns to be regrouped.

        Returns
        -------
        pd.DataFrame
            Returns DataFrame with regrouped variable as category datatype.
        """
        self.fit(X=X, y=y, columns=columns)

        X_tr = self.transform(X=X, columns=columns)

        return X_tr

    def _replaceMissings(self, X: pd.DataFrame,
                         column: str,
                         replace_with: str = 'Missing') -> pd.DataFrame:
        """
        Method replaces missing and empty cells with `Missing` (default) in
        a pd.DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe where a value will be replaced if empty or nan.

        column : str
            Column to be analyzed for missings.

        replace_with : str default='Missing'
            String to replace the missings.

        Raises
        ------
        ValueError
            In case input column is not a string.

        Returns
        -------
        pd.DataFrame
            Modified dataframe with replaced missings.
        """
        if X[column].dtype != 'O' or X[column].dtype != 'object':
            raise TypeError("column {} must be a string".format(column))

        X[column].fillna(replace_with, inplace=True)
        X[column] = X[column].astype(str).str.strip()
        X[column].replace('', replace_with, inplace=True)

        return X

    def _removeCategories(self, X: pd.DataFrame,
                          y: pd.Series,
                          column: str,
                          threshold: int = 5) -> np.ndarray:
        """
        Method removes category which fail to meet certain condition

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe with columns to be inspected for group removal.

        y : pd.Series
            Series with target.

        column : str
            Column to be analyzed group removal.

        threshold : int default=5
            Threshold for group removal.

        Returns
        -------
        np.ndarray
            Numpy array with groups to be kept.
        """
        category_cnts = pd.DataFrame(X.groupby(column)[column].count())
        train_inc = y.mean()
        factor = max(train_inc, 1-train_inc)
        keep_categories = category_cnts.where((category_cnts*factor) >
                                              threshold)

        return np.array(keep_categories.index.tolist())

    @staticmethod
    def _get_categorical_columns(data: pd.DataFrame) -> list:
        """Get the columns containing categorical data
        (dtype "object" or "category")

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe from which categorical variables
                will be extracted.

        Returns
        -------
        list
            List of column names containing categorical data.
        """
        object_columns = data.dtypes[data.dtypes == object].index
        categorical_columns = data.dtypes[data.dtypes == "category"].index

        return list(set(object_columns).union(set(categorical_columns)))
