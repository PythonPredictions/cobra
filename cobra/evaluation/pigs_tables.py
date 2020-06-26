# third party imports
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns

import cobra.utils as utils


def generate_pig_tables(data: pd.DataFrame,
                        id_column_name: str,
                        target_column_name: str,
                        preprocessed_predictors: list) -> pd.DataFrame:
    """Compute PIG tables for all predictors in preprocessed_predictors. The
    output is a DataFrame with columns "variable", "label", "pop_size",
    "avg_incidence" and "incidence"

    Parameters
    ----------
    data : pd.DataFrame
        basetable to compute PIG tables of
    id_column_name : str
        column name of the id (e.g. customernumber)
    target_column_name : str
        column name of the target
    predictors: list
        list of preprocessed predictor names

    Returns
    -------
    pd.DataFrame
        DataFrame containing a PIG table for all predictors
    """

    pigs = [compute_pig_table(data, column_name, target_column_name,
                              id_column_name)
            for column_name in sorted(preprocessed_predictors)
            if column_name not in [id_column_name, target_column_name]]

    output = pd.concat(pigs)

    return output


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
    res["variable"] = utils.clean_predictor_name(column_name)
    res["avg_incidence"] = avg_incidence
    res["pop_size"] = res["pop_size"]/len(data.index)

    # make sure to always return the data with the proper column order
    column_order = ["variable", "label", "pop_size",
                    "avg_incidence", "incidence"]

    return res[column_order]


def plot_pig_graph(pig_table: pd.DataFrame,
                   dim: tuple=(12, 8),
                   save_path: str=None):
    """Create the Predictor Insights Graphs from a PIG table

    Parameters
    ----------
    pig_table : pd.DataFrame
        Description
    dim : tuple, optional
        Tuple with width and lentgh of the plot
    save_path : str, optional
        path to store the plot on disk
    """
    pass
