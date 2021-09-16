"""
This class implements the Python Prediction's way of dealing with
categorical data preprocessing. There are three steps involved here:

- An optional regrouping of the different categories based on category size
  and significance of the category w.r.t. the target
- Missing value replacement with the additional category ``Missing``
- Change of dtype to ``category`` (could potentially lead to memory
  optimization)

Authors:
- Geert Verstraeten (methodology)
- Jan Benisek (implementation)
- Matthias Roels (implementation)
"""

# standard lib imports
import re
from typing import Optional
import logging

# third party imports
import numpy as np
import pandas as pd
from scipy import stats
from tqdm.auto import tqdm
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

log = logging.getLogger(__name__)


class CategoricalDataProcessor(BaseEstimator):
    """Regroups the categories of categorical variables based on significance
    with target variable.

    Attributes
    ----------
    category_size_threshold : int
        Minimal size of a category to keep it as a separate category.
    forced_categories : dict
        Map to prevent certain categories from being group into ``Other``
        for each column - dict of the form ``{col:[forced vars]}``.
    keep_missing : bool
        Whether or not to keep missing as a separate category.
    model_type : str
        Model type (``classification`` or ``regression``).
    p_value_threshold : float
        Significance threshold for regrouping.
    regroup : bool
        Whether or not to regroup categories.
    regroup_name : str
        New name of the non-significant regrouped variables
    scale_contingency_table : bool
        Whether contingency table should be scaled before chi^2.
    """

    valid_keys = ["model_type", "regroup", "regroup_name", "keep_missing",
                  "category_size_threshold", "p_value_threshold",
                  "scale_contingency_table", "forced_categories"]

    def __init__(self,
                 model_type: str = "classification",
                 regroup: bool = True,
                 regroup_name: str = "Other",
                 keep_missing: bool = True,
                 category_size_threshold: int = 5,
                 p_value_threshold: float = 0.001,
                 scale_contingency_table: bool = True,
                 forced_categories: dict = {}):
        
        if model_type not in ["classification", "regression"]:
            raise ValueError("An unexpected model_type was provided. A valid model_type is either 'classification' or 'regression'.")

        self.model_type = model_type
        self.regroup = regroup
        self.regroup_name = regroup_name
        self.keep_missing = keep_missing
        self.category_size_threshold = category_size_threshold
        self.p_value_threshold = p_value_threshold
        self.scale_contingency_table = scale_contingency_table
        self.forced_categories = forced_categories

        # dict to store fitted output in
        self._cleaned_categories_by_column = {}

    def attributes_to_dict(self) -> dict:
        """Return the attributes of CategoricalDataProcessor as a dictionary

        Returns
        -------
        dict
            Contains the attributes of CategoricalDataProcessor instance with
            the attribute name as key
        """
        params = self.get_params()

        params["_cleaned_categories_by_column"] = {
            key: list(value)
            for key, value in self._cleaned_categories_by_column.items()
        }

        return params

    def set_attributes_from_dict(self, params: dict):
        """Set instance attributes from a dictionary of values with key the
        name of the attribute.

        Parameters
        ----------
        params : dict
            Contains the attributes of CategoricalDataProcessor with their
            names as key.

        Raises
        ------
        ValueError
            In case _cleaned_categories_by_column is not of type dict
        """
        _fitted_output = params.pop("_cleaned_categories_by_column", {})

        if type(_fitted_output) != dict:
            raise ValueError("_cleaned_categories_by_column is expected to "
                             "be a dict but is of type {} instead"
                             .format(type(_fitted_output)))

        # Clean out params dictionary to remove unknown keys (for safety!)
        params = {key: params[key] for key in params if key in self.valid_keys}

        # We cannot turn this method into a classmethod as we want to make use
        # of the following method from BaseEstimator:
        self.set_params(**params)

        self._cleaned_categories_by_column = {
            key: set(value) for key, value in _fitted_output.items()
        }

        return self

    def fit(self, data: pd.DataFrame, column_names: list,
            target_column: str):
        """Fit the CategoricalDataProcessor

        Parameters
        ----------
        data : pd.DataFrame
            Data used to compute the mapping to encode the categorical
            variables with.
        column_names : list
            Columns of data to be processed.
        target_column : str
            Column name of the target.
        """

        if not self.regroup:
            # We do not need to fit anything if regroup is set to False!
            log.info("regroup was set to False, so no fitting is required")
            return None

        for column_name in tqdm(column_names, desc="Fitting category "
                                                   "regrouping..."):

            if column_name not in data.columns:
                log.warning("DataFrame has no column '{}', so it will be "
                            "skipped in fitting" .format(column_name))
                continue

            cleaned_cats = self._fit_column(data, column_name, target_column)

            # Remove forced categories
            forced_cats = self.forced_categories.get(column_name, set())
            cleaned_cats = cleaned_cats.union(forced_cats)

            # Add to _cleaned_categories_by_column for later use
            self._cleaned_categories_by_column[column_name] = cleaned_cats

    def _fit_column(self, data: pd.DataFrame, column_name: str,
                    target_column) -> set:
        """Compute which categories to regroup into "Other"
        for a particular column

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
        model_type = self.model_type

        if len(data[column_name].unique()) == 1:
            log.warning(f"Predictor {column_name} is constant"
                        " and will be ignored in computation.")
            return set(data[column_name].unique())

        y = data[target_column]
        if model_type == "classification":
            incidence = y.mean()
        else:
            incidence = None

        combined_categories = set()

        # replace missings and get unique categories as a list
        X = (CategoricalDataProcessor
             ._replace_missings(data[column_name])
             .astype(object))

        unique_categories = list(X.unique())

        # do not merge categories in case of dummies, i.e. 0 and 1
        # (and possibly "Missing")
        if (len(unique_categories) == 2
            or (len(unique_categories) == 3
                and "Missing" in unique_categories)):
            return set(unique_categories)

        # get small categories and add them to the merged category list
        # does not apply incidence factor when model_type = "regression"
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
                                      model_type,
                                      self.scale_contingency_table))

            # if not significant, add it to the list
            if pval > self.p_value_threshold:
                combined_categories.add(category)

        # Remove missing category from combined_categories if required
        if self.keep_missing:
            combined_categories.discard("Missing")

        return set(unique_categories).difference(combined_categories)

    def transform(self, data: pd.DataFrame,
                  column_names: list) -> pd.DataFrame:
        """Transform the data

        Parameters
        ----------
        data : pd.DataFrame
            data used to compute the mapping to encode the categorical
            variables with.
        column_names : list
            Columns of data to be processed

        Returns
        -------
        pd.DataFrame
            data with additional transformed variables
        """

        if self.regroup and len(self._cleaned_categories_by_column) == 0:
            msg = ("{} instance is not fitted yet. Call 'fit' with "
                   "appropriate arguments before using this method.")

            raise NotFittedError(msg.format(self.__class__.__name__))

        for column_name in column_names:

            if column_name not in data.columns:
                log.warning("Unknown column '{}' will be skipped"
                            .format(column_name))
                continue

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
        data.loc[:, column_name_clean] = data[column_name].astype(object)

        # Fill missings first
        data.loc[:, column_name_clean] = (CategoricalDataProcessor
                                          ._replace_missings(
                                              data,
                                              column_name_clean
                                              ))

        if self.regroup:
            categories = self._cleaned_categories_by_column.get(column_name)

            if not categories:
                # Log warning if categories is None, which indicates it is
                # not in fitted output
                if categories is None:
                    log.warning("Column '{}' is not in fitted output "
                                "and will be skipped".format(column_name))
                return data

            data.loc[:, column_name_clean] = (CategoricalDataProcessor
                                              ._replace_categories(
                                                  data[column_name_clean],
                                                  categories,
                                                  self.regroup_name))

        # change data to categorical
        data.loc[:, column_name_clean] = (data[column_name_clean]
                                          .astype("category"))

        return data

    def fit_transform(self, data: pd.DataFrame, column_names: list,
                      target_column: str) -> pd.DataFrame:
        """Fits to data, then transform it

        Parameters
        ----------
        data : pd.DataFrame
            data used to compute the mapping to encode the categorical
            variables with.
        column_names : list
            Columns of data to be processed
        target_column : str
            Column name of the target

        Returns
        -------
        pd.DataFrame
            data with additional transformed variables
        """

        self.fit(data, column_names, target_column)
        return self.transform(data, column_names)

    @staticmethod
    def _get_small_categories(predictor_series: pd.Series,
                              incidence: float,
                              category_size_threshold: int) -> set:
        """Fetch categories with a size below a certain threshold.
        Note that we use an additional weighting with the overall incidence.

        Parameters
        ----------
        predictor_series : pd.Series
            Variables data.
        incidence : float
            Global train incidence.
        category_size_threshold : int
            Minimal size of a category to keep it as a separate category.

        Returns
        -------
        set
            List a categories with a count below a certain threshold.
        """
        category_counts = predictor_series.groupby(predictor_series).size()
        if incidence is not None:
            factor = max(incidence, 1 - incidence)
        else:
            factor = 1

        # Get all categories with a count below a threshold
        bool_mask = (category_counts*factor) <= category_size_threshold
        return set(category_counts[bool_mask].index.tolist())

    @staticmethod
    def _replace_missings(data: pd.DataFrame,
                          column_names: Optional[list] = None) -> pd.DataFrame:
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
                         model_type: str,
                         scale_contingency_table: bool) -> float:
        """Calculates p-value in order to evaluate whether category of
        interest is significantly different from the rest of the
        categories, given the target variable.

        In case model_type is "classification", chi-squared test based on a contingency table.
        In case model_type is "regression", Kruskal-Wallis test.

        Parameters
        ----------
        X : pd.Series
            Variables data.
        y : pd.Series
            Target data.
        category : str
            Category for which we carry out the test.
        model_type : str
            Model type (``classification`` or ``regression``).
        scale_contingency_table : bool
            Whether we scale contingency table with incidence rate.
            Only used when model_type = "classification".

        Returns
        -------
        float
            p-value of applied statistical test
        """
        df = pd.concat([X, y], axis=1)
        df.columns = ["X", "y"]
        df["other_categories"] = np.where(X == category, 0, 1)

        if model_type == "classification":
            contingency_table = pd.crosstab(index=df["other_categories"], columns=df["y"],
                                            margins=False)

            # if true, we scale the "other" categories
            if scale_contingency_table:
                size_other_cats = contingency_table.iloc[1].sum()
                incidence_mean = y.mean()

                contingency_table.iloc[1, 0] = (1-incidence_mean) * size_other_cats
                contingency_table.iloc[1, 1] = incidence_mean * size_other_cats
                contingency_table = contingency_table.values.astype(np.int64)

            pval = stats.chi2_contingency(contingency_table, correction=False)[1]

        elif model_type == "regression":
            pval = stats.kruskal(df.y[df.other_categories == 0],
                                 df.y[df.other_categories == 1])[1]

        return pval

    @staticmethod
    def _replace_categories(data: pd.Series, categories: set,
                            replace_with: str) -> pd.Series:
        """replace categories in set with "Other" and transform the remaining
        categories to strings to avoid type errors later on in the pipeline

        Parameters
        ----------
        data : pd.Series
            Dataset which contains the variable to be replaced
        categories : set
            Cleaned categories.
        replace_with: str
            String to be used as replacement for category.

        Returns
        -------
        pd.Series
            Series with replaced categories
        """
        return data.apply(
            lambda x: str(x) if x in categories else replace_with)
