# standard lib imports
import inspect
import time
import math
import logging
from random import shuffle
from datetime import datetime

# third party imports
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

# custom imports
from cobra.preprocessing import CategoricalDataProcessor
from cobra.preprocessing import KBinsDiscretizer
from cobra.preprocessing import TargetEncoder

log = logging.getLogger(__name__)


class PreProcessor(BaseEstimator):
    """This class implements a so-called facade pattern to define a
    higher-level interface to work with the CategoricalDataProcessor,
    KBinsDiscretizer and TargetEncoder classes, so that their fit and transform
    methods are called in the correct order.

    Additionally, it provides methods such as (de)serialization to/from JSON
    so that preprocessing pipelines can be stored and reloaded, example for scoring.

    We refer to the README of the GitHub repository for more background information
    on the preprocessing methodology.

    Attributes
    ----------
    categorical_data_processor : CategoricalDataProcessor
        Instance of CategoricalDataProcessor to do the preprocessing of
        categorical variables.
    discretizer : KBinsDiscretizer
        Instance of KBinsDiscretizer to do the preprocessing of continuous
        variables by means of discretization.
    target_encoder : TargetEncoder
        Instance of TargetEncoder to do the incidence replacement.
    is_fitted : bool
        Whether or not object is yet fit.
    model_type : str
        The model_type variable as specified in CategoricalDataProcessor
        (``classification`` or ``regression``).
    """

    def __init__(
        self,
        categorical_data_processor: CategoricalDataProcessor,
        discretizer: KBinsDiscretizer,
        target_encoder: TargetEncoder,
        is_fitted: bool = False,
    ):

        self.categorical_data_processor = categorical_data_processor
        self.discretizer = discretizer
        self.target_encoder = target_encoder

        self.is_fitted = is_fitted

        self.model_type = categorical_data_processor.model_type

    @classmethod
    def from_params(
        cls,
        model_type: str = "classification",
        n_bins: int = 10,
        strategy: str = "quantile",
        closed: str = "right",
        auto_adapt_bins: bool = False,
        starting_precision: int = 0,
        label_format: str = "{} - {}",
        change_endpoint_format: bool = False,
        regroup: bool = True,
        regroup_name: str = "Other",
        keep_missing: bool = True,
        category_size_threshold: int = 5,
        p_value_threshold: float = 0.001,
        scale_contingency_table: bool = True,
        forced_categories: dict = {},
        weight: float = 0.0,
        imputation_strategy: str = "mean",
    ):
        """Constructor to instantiate PreProcessor from all the parameters
        that can be set in all its required (attribute) classes
        along with good default values.

        Parameters
        ----------
        model_type : str
            Model type (``classification`` or ``regression``).
        n_bins : int, optional
            Number of bins to produce. Raises ValueError if ``n_bins < 2``.
        strategy : str, optional
            Binning strategy. Currently only ``uniform`` and ``quantile``
            e.g. equifrequency is supported.
        closed : str, optional
            Whether to close the bins (intervals) from the left or right.
        auto_adapt_bins : bool, optional
            Reduces the number of bins (starting from n_bins) as a function of
            the number of missings.
        starting_precision : int, optional
            Initial precision for the bin edges to start from,
            can also be negative. Given a list of bin edges, the class will
            automatically choose the minimal precision required to have proper
            bins e.g. ``[5.5555, 5.5744, ...]`` will be rounded
            to ``[5.56, 5.57, ...]``. In case of a negative number, an attempt
            will be made to round up the numbers of the bin edges
            e.g. ``5.55 -> 10``, ``146 -> 100``, ...
        label_format : str, optional
            Format string to display the bin labels
            e.g. ``min - max``, ``(min, max]``, ...
        change_endpoint_format : bool, optional
            Whether or not to change the format of the lower and upper bins
            into ``< x`` and ``> y`` resp.
        regroup : bool
            Whether or not to regroup categories.
        regroup_name : str
            New name of the non-significant regrouped variables.
        keep_missing : bool
            Whether or not to keep missing as a separate category.
        category_size_threshold : int
            All categories with a size (corrected for incidence if applicable)
            in the training set above this threshold are kept as a separate category,
            if statistical significance w.r.t. target is detected. Remaining
            categories are converted into ``Other`` (or else, cf. regroup_name).
        p_value_threshold : float
            Significance threshold for regrouping.
        forced_categories : dict
            Map to prevent certain categories from being grouped into ``Other``
            for each column - dict of the form ``{col:[forced vars]}``.
        scale_contingency_table : bool
            Whether contingency table should be scaled before chi^2.
        weight : float, optional
            Smoothing parameters (non-negative). The higher the value of the
            parameter, the bigger the contribution of the overall mean.
            When set to zero, there is no smoothing (e.g. the pure target incidence is used).
        imputation_strategy : str, optional
            Valid imputation strategies = mean, median, min or max
            In case there is a particular column which contains new categories,
            the encoding will lead to NULL values which should be imputed.
            For more information about how the imputation works go see the documentation of the 
            :class:`cobra.preprocessing.TargetEncoder` 

        Returns
        -------
        PreProcessor
            Class encapsulating CategoricalDataProcessor,
            KBinsDiscretizer, and TargetEncoder instances.
        """
        categorical_data_processor = CategoricalDataProcessor(
            model_type,
            regroup,
            regroup_name,
            keep_missing,
            category_size_threshold,
            p_value_threshold,
            scale_contingency_table,
            forced_categories,
        )

        discretizer = KBinsDiscretizer(
            n_bins,
            strategy,
            closed,
            auto_adapt_bins,
            starting_precision,
            label_format,
            change_endpoint_format,
        )

        target_encoder = TargetEncoder(weight, imputation_strategy)

        return cls(categorical_data_processor, discretizer, target_encoder)

    @classmethod
    def from_pipeline(cls, pipeline: dict):
        """Constructor to instantiate PreProcessor from a (fitted) pipeline
        which was stored as a JSON file and passed to this function as a dict.

        Parameters
        ----------
        pipeline : dict
            The (fitted) pipeline as a dictionary.

        Returns
        -------
        PreProcessor
            Instance of PreProcessor instantiated from a stored pipeline.

        Raises
        ------
        ValueError
            If the loaded pipeline does not have all required parameters
            and no others.
        """

        if not PreProcessor._is_valid_pipeline(pipeline):
            raise ValueError(
                "Invalid pipeline, as it does not "
                "contain all and only the required parameters."
            )

        categorical_data_processor = CategoricalDataProcessor()
        categorical_data_processor.set_attributes_from_dict(
            pipeline["categorical_data_processor"]
        )
        # model_type = categorical_data_processor.model_type

        discretizer = KBinsDiscretizer()
        discretizer.set_attributes_from_dict(pipeline["discretizer"])

        target_encoder = TargetEncoder()
        target_encoder.set_attributes_from_dict(pipeline["target_encoder"])

        return cls(
            categorical_data_processor,
            discretizer,
            target_encoder,
            is_fitted=pipeline["_is_fitted"],
        )
        
    def get_continuous_and_discrete_columns(
        self, df: pd.DataFrame, id_col_name: str, target_column_name: str
    ) -> tuple:
        """Filters out the continuous and discrete  variables out of a dataframe and returns a tuple containing lists of column names
        It assumes that numerical columns with less than or equal to 10 different values are categorical

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame that you want to divide in discrete and continuous variables
        id_col_name : str
            column name of the id column, can be None
        target_column_name : str
            column name of the target column

        Returns
        -------
        tuple
            tuple containing 2 lists of column names. (continuous_vars, discrete_vars)
        """
        if id_col_name == None:
            log.warning(
                "id_col_name is equal to None. If there is no id column ignore this warning"
            )

        # find continuous_vars and discrete_vars in the dateframe
        col_dtypes = df.dtypes
        discrete_vars = [
            col
            for col in col_dtypes[col_dtypes == object].index.tolist()
            if col not in [id_col_name, target_column_name]
        ]

        for col in df.columns:
            if col not in discrete_vars and col not in [
                id_col_name,
                target_column_name,
            ]:  # omit discrete because a string, and target
                val_counts = df[col].nunique()
                if (
                    val_counts > 1 and val_counts <= 10
                ):  # the column contains less than 10 different values
                    discrete_vars.append(col)

        continuous_vars = list(
            set(df.columns)
            - set(discrete_vars)
            - set([id_col_name, target_column_name])
        )
        log.warning(
            f"""Cobra automaticaly assumes that following variables are 
            discrete: {discrete_vars}
            continuous: {continuous_vars}
            If you want to change this behaviour you can specify the discrete/continuous variables yourself with the continuous_vars and discrete_vars keywords.
            It assumes that numerical columns with less than or equal to 10 different values are categorical"""
        )
        return continuous_vars, discrete_vars

    def fit(
        self,
        train_data: pd.DataFrame,
        continuous_vars: list,
        discrete_vars: list,
        target_column_name: str,
        id_col_name: str = None,
    ):
        """Fit the data to the preprocessing pipeline.
        If you put continuous_vars and target_vars equal to `None` and give the id_col_name Cobra will guess which variables are continuous and which are not.

        Parameters
        ----------
        train_data : pd.DataFrame
            Data to be preprocessed.
        continuous_vars : list | None
            List of continuous variables, can be None.
        discrete_vars : list | None
            List of discrete variables, can be None.
        target_column_name : str
            Column name of the target.
        id_col_name : str, optional
            _description_, by default None
        """
        if not (continuous_vars and discrete_vars):
            continuous_vars, discrete_vars = self.get_continuous_and_discrete_columns(
                df=train_data,
                id_col_name=id_col_name,
                target_column_name=target_column_name,
            )

        # get list of all variables
        preprocessed_variable_names = PreProcessor._get_variable_list(
            continuous_vars, discrete_vars
        )

        log.info("Starting to fit pipeline")
        start = time.time()

        # Ensure to operate on separate copy of data
        train_data = train_data.copy()

        # drop NAN columns if they exist
        train_data = (
            PreProcessor._check_nan_columns_and_drop_columns_containing_only_nan(
                train_data
            )
        )

        # Fit discretizer, categorical preprocessor & target encoder
        # Note that in order to fit target_encoder, we first have to transform
        # the data using the fitted discretizer & categorical_data_processor
        if continuous_vars:
            begin = time.time()
            self.discretizer.fit(train_data, continuous_vars)
            log.info(
                "Fitting KBinsDiscretizer took {} seconds".format(time.time() - begin)
            )

            train_data = self.discretizer.transform(train_data, continuous_vars)
        if discrete_vars:
            begin = time.time()
            self.categorical_data_processor.fit(
                train_data, discrete_vars, target_column_name
            )
            log.info(
                "Fitting categorical_data_processor class took {} seconds".format(
                    time.time() - begin
                )
            )

            train_data = self.categorical_data_processor.transform(
                train_data, discrete_vars
            )

        begin = time.time()
        self.target_encoder.fit(
            train_data, preprocessed_variable_names, target_column_name
        )
        log.info("Fitting TargetEncoder took {} seconds".format(time.time() - begin))

        self.is_fitted = True  # set fitted boolean to True

        log.info("Fitting pipeline took {} seconds".format(time.time() - start))

    def transform(
        self, data: pd.DataFrame, continuous_vars: list, discrete_vars: list
    ) -> pd.DataFrame:
        """Transform the data by applying the preprocessing pipeline.

        Parameters
        ----------
        data : pd.DataFrame
            Data to be preprocessed.
        continuous_vars : list
            List of continuous variables.
        discrete_vars : list
            List of discrete variables.

        Returns
        -------
        pd.DataFrame
            Transformed (preprocessed) data.

        Raises
        ------
        NotFittedError
            In case PreProcessor was not fitted first.
        """

        start = time.time()

        # Ensure to operate on separate copy of data
        data = data.copy()

        if not self.is_fitted:
            msg = (
                "This {} instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this method."
            )

            raise NotFittedError(msg.format(self.__class__.__name__))

        preprocessed_variable_names = PreProcessor._get_variable_list(
            continuous_vars, discrete_vars
        )

        if continuous_vars:
            data = self.discretizer.transform(data, continuous_vars)

        if discrete_vars:
            data = self.categorical_data_processor.transform(data, discrete_vars)

        data = self.target_encoder.transform(data, preprocessed_variable_names)

        log.info("Transforming data took {} seconds".format(time.time() - start))

        return data

    def fit_transform(
        self,
        train_data: pd.DataFrame,
        continuous_vars: list,
        discrete_vars: list,
        target_column_name: str,
        id_col_name: str = None,
    ) -> pd.DataFrame:

        """Fit preprocessing pipeline and transform the data.
        If you put continuous_vars and target_vars equal to `None` and give the id_col_name Cobra will guess which variables are continuous and which are not.

        Parameters
        ----------
        train_data : pd.DataFrame
            Data to be preprocessed
        continuous_vars : list
            List of continuous variables, can be None.
        discrete_vars : list
            List of discrete variables, can be None.
        target_column_name : str
            Column name of the target.
        id_col_name : str, optional
            _description_, by default None

        Returns
        -------
        pd.DataFrame
            Transformed (preprocessed) data.
        """
        if not (continuous_vars and discrete_vars) and id_col_name:
            continuous_vars, discrete_vars = self.get_continuous_and_discrete_columns(
                df=train_data,
                id_col_name=id_col_name,
                target_column_name=target_column_name,
            )
        self.fit(
            train_data, continuous_vars, discrete_vars, target_column_name, id_col_name
        )

        return self.transform(train_data, continuous_vars, discrete_vars)

    @staticmethod
    def train_selection_validation_split(
        data: pd.DataFrame,
        train_prop: float = 0.6,
        selection_prop: float = 0.2,
        validation_prop: float = 0.2,
    ) -> pd.DataFrame:
        """Adds `split` column with train/selection/validation values
        to the dataset.

        Train set = data on which the model is trained and on which the encoding is based.
        Selection set = data used for univariate and forward feature selection. Often called the validation set.
        Validation set = data that generates the final performance metrics. Often called the test set.

        Parameters
        ----------
        data : pd.DataFrame
            Input dataset to split into train-selection and validation sets.
        train_prop : float, optional
            Percentage data to put in train set.
        selection_prop : float, optional
            Percentage data to put in selection set.
        validation_prop : float, optional
            Percentage data to put in validation set.

        Returns
        -------
        pd.DataFrame
            DataFrame with additional split column.
        """
        if not math.isclose(train_prop + selection_prop + validation_prop, 1.0):
            raise ValueError(
                "The sum of train_prop, selection_prop and "
                "validation_prop must be 1.0."
            )

        if train_prop == 0.0:
            raise ValueError("train_prop cannot be zero!")

        if selection_prop == 0.0:
            raise ValueError("selection_prop cannot be zero!")

        nrows = data.shape[0]
        size_train = int(train_prop * nrows)
        size_select = int(selection_prop * nrows)
        size_valid = int(validation_prop * nrows)
        correction = nrows - (size_train + size_select + size_valid)

        split = (
            ["train"] * size_train
            + ["train"] * correction
            + ["selection"] * size_select
            + ["validation"] * size_valid
        )

        shuffle(split)

        data["split"] = split

        return data

    def serialize_pipeline(self) -> dict:
        """Serialize the preprocessing pipeline by writing all its required
        parameters to a dictionary to later store it as a JSON file.

        Returns
        -------
        dict
            Return the pipeline as a dictionary.
        """
        pipeline = {
            "metadata": {"timestamp": datetime.now().strftime("%d/%m/%Y %H:%M:%S")}
        }

        pipeline[
            "categorical_data_processor"
        ] = self.categorical_data_processor.attributes_to_dict()

        pipeline["discretizer"] = self.discretizer.attributes_to_dict()
        pipeline["target_encoder"] = self.target_encoder.attributes_to_dict()

        pipeline["_is_fitted"] = True

        return pipeline

    @staticmethod
    def _is_valid_pipeline(pipeline: dict) -> bool:
        """Validate the loaded pipeline by checking if all required parameters
        are present (and no others!).

        Parameters
        ----------
        pipeline : dict
            Loaded pipeline from JSON file.
        """
        keys = inspect.getfullargspec(PreProcessor.from_params).args
        valid_keys = set(
            [key for key in keys if key not in ["cls", "serialization_path"]]
        )

        input_keys = set()
        for key in pipeline:
            if key in ["categorical_data_processor", "discretizer", "target_encoder"]:
                input_keys = input_keys.union(set(pipeline[key].keys()))
            elif key != "metadata":
                input_keys.add(key)

        input_keys = sorted(list(input_keys))
        input_keys = [key for key in input_keys if not key.startswith("_")]

        return sorted(list(valid_keys)) == sorted(list(input_keys))

    @staticmethod
    def _get_variable_list(continuous_vars: list, discrete_vars: list) -> list:
        """Merge lists of continuous_vars and discrete_vars and add suffix
        "_bin" resp. "_processed" to the predictors.

        Parameters
        ----------
        continuous_vars : list
            List of continuous variables.
        discrete_vars : list
            List of discrete variables.

        Returns
        -------
        list
            Merged list of predictors with proper suffixes added.

        Raises
        ------
        ValueError
            In case both lists are empty.
        """
        var_list = [col + "_processed" for col in discrete_vars] + [
            col + "_bin" for col in continuous_vars
        ]

        if not var_list:
            raise ValueError("Variable var_list is None or empty list.")

        return var_list

    def _check_nan_columns_and_drop_columns_containing_only_nan(
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Checks how much missing values are in the dataframe and drops columns that contain only missing values.
        It also logs an error message displaying the percentage of missing values in the different columns
        (columns are only displayed if they contain a missing values)

        Parameters
        ----------
        data : pd.DataFrame
            Data that should be checked for columns that contain only missing values

        Returns
        -------
        pd.DataFrame
            Data without columns containing only missing values
        """
        # Ensure to operate on separate copy of data
        data = data.copy()

        # Check how much NaN values are in each variable
        # and output a warning if a variable has more than 0% of missing values

        perc_na = data.isna().mean() * 100

        if not perc_na[perc_na > 0].empty:
            logging.warning(
                "\nPercentage of missing values per variable:\n"
                + perc_na[perc_na > 0]
                .round(2)
                .to_string(float_format=lambda x: str(x) + "%")
            )

        # drop variables that have only missing values
        to_drop = [
            perc_na.index[i]
            for i, percentage in enumerate(perc_na)
            if percentage == 100
        ]

        if to_drop:
            data = data.drop(to_drop, axis=1)
            logging.warning(
                f"Following variables contain only missing values and were dropped: {to_drop}"
            )

        return data
