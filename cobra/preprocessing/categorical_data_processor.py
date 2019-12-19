"""
This class implements the Python Prediction's way of dealing with
categorical data preprocessing. There are two steps involved here:
- An optional regrouping of the different categories based on category size
  and significance of the category
- Missing value replacement with the additional category "Missing"

Authors:
- Geert Verstraeten (methodology)
- Jan Benisek (implementation)
- Matthias Roels (implementation)
"""
# standard lib imports
import re
from typing import Optional

import logging
log = logging.getLogger(__name__)

# third party imports
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError


class CategoricalDataProcessor(BaseEstimator):
    """
    Regroups categories in categorical variables based on significance
    with target variable.

    Attributes
    ----------
    category_size_threshold : int, optional
        minimal size of a category to keep it as a separate category
    forced_categories : dict, optional
        Map to prevent certain categories from being group into "Other"
        for each colum - dict of the form {col:[forced vars]}.
    keep_missing : bool
        Whether or not to keep missing as a separate category
    p_value_threshold : float
        Significance threshold for regroupping.
    regroup : bool
        Whether or not to regroup categories
    regroup_name : str
        New name of the non-significant regrouped variables
    scale_contingency_table : bool
        Whether contingency table should be scaled before chi^2.'
    """

    def __init__(self, regroup: bool=True, regroup_name: str="Other",
                 keep_missing: bool=True,
                 category_size_threshold: Optional[int]=None,
                 p_value_threshold: float=0.001,
                 scale_contingency_table: bool=True,
                 forced_categories: Optional[dict]=None):

        self.regroup = regroup
        self.regroup_name = regroup_name
        self.keep_missing = keep_missing
        self.category_size_threshold = category_size_threshold
        self.p_value_threshold = p_value_threshold
        self.scale_contingency_table = scale_contingency_table
        self.forced_categories = forced_categories

        # dict to store fitted output in
        self._combined_categories_by_column = {}

    def fit(self, data: pd.DataFrame, column_names: list,
            target_column: str):
        """Fit the CategoricalDataProcessor

        Parameters
        ----------
        data : pd.DataFrame
            data used to compute the mapping to encode the categorical
            variables with.
        column_names : list
            Columns of data to be processed
        target_column : str
            Column name of the target
        """

        if not self.regroup:
            # We do not need to fit anything if regroup is set to False!
            log.info("regroup was set to False, so no fitting is required")
            return None

        for column_name in column_names:

            if column_name not in data.columns:
                log.warning("DataFrame has no column '{}', so it will be "
                            "skipped in fitting" .format(column_name))
                continue

            combined_cats = self._fit_column(data, column_name, target_column)

            # Add to _combined_categories_by_column for later use
            self._combined_categories_by_column[column_name] = combined_cats

    def _fit_column(self, data: pd.DataFrame, column_name: str,
                    target_column) -> list:
        """Compute which categories to regroup into "Other" for a particular
        column

        Parameters
        ----------
        data : pd.DataFrame
            Description
        column_name : str
            Description

        Returns
        -------
        list
            list of categories to combine into a category "Other"
        """

        X = data[column_name]
        y = data[target_column]
        incidence = y.mean()

        combined_categories = set()

        # replace missings and get unique categories as a list
        X = CategoricalDataProcessor._replace_missings(X)
        unique_categories = list(X.unique())

        # get small categories and add them to the merged category list
        small_categories = (CategoricalDataProcessor
                            ._get_small_categories(
                                X,
                                incidence,
                                self.category_size_threshold))
        combined_categories = combined_categories.union(small_categories)

        for category in unique_categories:
            if category in small_categories:
                continue

            pval = (CategoricalDataProcessor
                    ._compute_p_value(X, y, category,
                                      self.scale_contingency_table))

            # if not significant, add it to the list
            if pval > self.p_value_threshold:
                combined_categories.add(category)

        # Remove missing category from combined_categories if required
        if self.keep_missing:
            combined_categories.discard("Missing")

        return combined_categories

    def transform(self, data: pd.DataFrame,
                  column_names: list) -> pd.DataFrame:
        """Summary

        Parameters
        ----------
        data : pd.DataFrame
            Data to be discretized
        column_names : list
            Columns of data to be discretized

        Returns
        -------
        pd.DataFrame
            data with additional discretized variables
        """

        if self.regroup and len(self._combined_categories_by_column) == 0:
            msg = ("{} instance is not fitted yet. Call 'fit' with "
                   "appropriate arguments before using this method.")

            raise NotFittedError(msg.format(self.__class__.__name__))

        for column_name in column_names:

            data = self._transform_column(data, column_name)

        return data

    def _transform_column(self, data: pd.DataFrame,
                          column_name: str) -> pd.DataFrame:
        """Given a DataFrame, a column name and a list of categories to
        combine, create an additional column which combines these categories
        into "Other"

        Parameters
        ----------
        data : pd.DataFrame
            Original data to be tranformed
        column_name : str
            name of the column to transform

        Returns
        -------
        pd.DataFrame
            original DataFrame with an added processed column
        """

        column_name_clean = column_name + "_processed"
        data[column_name_clean] = data[column_name]

        # Fill missings first
        data[column_name_clean] = (CategoricalDataProcessor
                                   ._replace_missings(data,
                                                      column_name_clean))

        if self.regroup:
            categories = self._combined_categories_by_column.get(column_name)

            data[column_name_clean] = (CategoricalDataProcessor
                                       ._replace_categories(
                                           data[column_name_clean],
                                           categories))

        # change data to categorical
        data[column_name_clean] = data[column_name_clean].astype("category")

        return data

    def fit_transform(self, data: pd.DataFrame,
                      column_names: list) -> pd.DataFrame:
        """Summary

        Parameters
        ----------
        data : pd.DataFrame
            Data to be discretized
        column_names : list
            Columns of data to be discretized

        Returns
        -------
        pd.DataFrame
            data with additional discretized variables
        """
        self.fit(data, column_names)
        return self.transform(data, column_names)

    @staticmethod
    def _get_small_categories(predictor_series: pd.Series,
                              incidence: float,
                              category_size_threshold: int) -> set:
        """Fetch categories with a size below a certain threshold.
        Note that we use an additional weighting with the overall incidence

        Parameters
        ----------
        predictor_series : pd.Series
            Description
        incidence : float
            global train incidence
        category_size_threshold : int
            minimal size of a category to keep it as a separate category

        Returns
        -------
        set
            List a categories with a count below a certain threshold
        """
        category_counts = predictor_series.groupby(predictor_series).size()
        factor = max(incidence, 1 - incidence)

        # Get all categories with a count below a threshold
        bool_mask = (category_counts*factor) <= category_size_threshold
        return set(category_counts[bool_mask].index.tolist())

    @staticmethod
    def _replace_missings(data: pd.DataFrame,
                          column_names: Optional[list]=None) -> pd.DataFrame:
        """Replace missing values (incl empty strings)

        Parameters
        ----------
        data : pd.DataFrame
            data to replace missings in
        column_names: list, optional
            list of predictors to replace missings in

        Returns
        -------
        list
            list of unique values in the data
        """
        # replace missings (incl. empty string)
        regex = re.compile("^\\s+|\\s+$")

        temp = None
        if column_names:
            temp = data[column_names]
        else:
            temp = data.copy()
        temp = temp.fillna("Missing")
        temp = temp.replace(regex, "")
        temp = temp.replace("", "Missing")

        return temp

    @staticmethod
    def _compute_p_value(X: pd.Series, y: pd.Series, category: str,
                         scale_contingency_table: bool) -> float:
        """Summary

        Parameters
        ----------
        X : pd.Series
            Description
        y : pd.Series
            Description
        category : str
            Description
        scale_contingency_table : bool
            Description

        Returns
        -------
        float
            Description
        """
        df = pd.concat([X, y], axis=1)
        df["other_categories"] = np.where(X == category, 0, 1)

        contigency_table = pd.crosstab(index=df['other_categories'], columns=y,
                                       margins=False)

        # if true, we scale the "other" categories
        if scale_contingency_table:
            size_other_cats = contigency_table.iloc[1].sum()
            incidence_mean = y.mean()

            contigency_table.iloc[1, 0] = (1-incidence_mean) * size_other_cats
            contigency_table.iloc[1, 1] = incidence_mean * size_other_cats
            contigency_table = contigency_table.values.astype(np.int64)

        return stats.chi2_contingency(contigency_table, correction=False)[1]

    @staticmethod
    def _replace_categories(data: pd.Series, categories: set) -> pd.Series:
        """replace categories in set with "Other"

        Parameters
        ----------
        data : pd.Series
            Description
        categories : set
            Description

        Returns
        -------
        pd.Series
            Description
        """
        return data.apply(lambda x: x if x not in categories else "Other")
