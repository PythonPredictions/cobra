# third party lib imports
import pandas as pd
# custom imports
import cobra.utils as utils
from cobra.preprocessing.kbins_discretizer import KBinsDiscretizer


def preprocess_categoricals(data: pd.DataFrame,
                            categorical_columns: list) -> pd.DataFrame:

    for column_name in categorical_columns:

        # change data to categorical
        data[column_name] = data[column_name].astype("category")

        # check for null values
        if data[column_name].isnull().sum() > 0:

            # Add an additional category
            data[column_name].cat.add_categories(["Missing"], inplace=True)

            # Replace NULL with "Missing"
            # Otherwise these will be ignored in groupby
            data[column_name].fillna("Missing", inplace=True)

    return data


def compute_pig_table(data: pd.DataFrame,
                      column_name: str,
                      target_column_name: str,
                      id_column_name: str) -> pd.DataFrame:
    """Compute the pig table of a given predictor for a given target

    Parameters
    ----------
    data : pd.DataFrame
        input data from which to compute the pig table
    column_name : str
        predictor name of which to compute the pig table
    target_column_name : str
        name of the target variable
    id_column_name : str
        name of the id column (used to count population size)

    Returns
    -------
    pd.DataFrame
        pig table as a DataFrame
    """
    avg_incidence = data[target_column_name].mean()

    # group by the binned variable, compute the incidence
    # (=mean of the target for the given bin) and compute the bin size
    # (e.g. COUNT(id_column_name)). After that, rename the columns
    res = (data.groupby(column_name)
               .agg({target_column_name: "mean", id_column_name: "size"})
               .reset_index()
               .rename(columns={column_name: "label",
                                target_column_name: "incidence",
                                id_column_name: "pop_size"}))

    # add the column name to a variable column
    # add the average incidence
    # replace population size by a percentage of total population
    res["variable"] = column_name
    res["avg_incidence"] = avg_incidence
    res["pop_size"] = res["pop_size"]/len(data.index)

    # make sure to always return the data with the proper column order
    column_order = ["variable", "label", "pop_size",
                    "avg_incidence", "incidence"]

    return res[column_order]


def generate_pig_tables(data: pd.DataFrame,
                        id_column_name: str,
                        target_column_name: str,
                        n_bins: int,
                        strategy: str,
                        label_format: str) -> pd.DataFrame:
    """Summary

    Parameters
    ----------
    data : pd.DataFrame
        basetable to compute PIG tables of
    id_column_name : str
        column name of the id (e.g. customernumber)
    target_column_name : str
        column name of the target
    n_bins : int
        Number of bins to produce after discretization
    strategy : str
        Binning strategy. Currently only "uniform" and "quantile"
        e.g. equifrequency is supported
    label_format : str
        format string to display the bin labels e.g. min - max, (min, max], ...

    Returns
    -------
    pd.DataFrame
        DataFrame containing a PIG table for all predictors
    """

    # Based on the data, get column names by datatype
    # threshold to decide whether a numeric column should be considered
    # a categorical variable (if the number of distinct values is smaller
    # or equal to the number of requested bins)
    categorical_threshold = n_bins
    columns_by_type = utils.get_column_datatypes(data, id_column_name,
                                                 target_column_name,
                                                 categorical_threshold)

    # process continuous variables
    discretizer = KBinsDiscretizer(n_bins=n_bins,
                                   strategy=strategy,
                                   label_format=label_format)

    # Transform the data
    data = discretizer.fit_transform(data,
                                     columns_by_type["numeric_variables"])

    # Process categorical and dummy variables
    categorical_vars = columns_by_type["categorical_variables"]
    dummy_vars = columns_by_type["dummy_variables"]
    relevant_columns = set(categorical_vars).union(set(dummy_vars))

    data = preprocess_categoricals(data, list(relevant_columns))

    # Get relevant columns, e.g. the ones that are transformed
    # into categorical dtypes by the preprocessing steps
    relevant_columns = set(data.dtypes[data.dtypes == "category"].index)

    pigs = [compute_pig_table(data, column_name, target_column_name,
                              id_column_name)
            for column_name in sorted(relevant_columns)
            if column_name not in [id_column_name, target_column_name]]

    output = pd.concat(pigs)

    return output
