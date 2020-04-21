import numpy as np
import pandas as pd


def get_column_datatypes(data: pd.DataFrame,
                         target_column_name: str=None,
                         id_column_name: str=None,
                         threshold_numeric_is_categorical: int=10) -> dict:
    """Get a list of column names by data type from a pandas DataFrame,
    excluding the id column and the target_column if provided

    Parameters
    ----------
    data : pd.DataFrame
        data to extract columns by type from
    target_column_name : str, optional
        Description
    id_column_name : str, optional
        Description
    threshold_numeric_is_categorical : int, optional
        Threshold to decide whether a numeric variable is categorical based
        on the number of unique values in that column

    Returns
    -------
    dict
        Description
    """

    # categorical vars
    vars_cat = (set(data.dtypes[data.dtypes == object].index)
                .union(set(data.dtypes[data.dtypes == "category"].index)))

    # Numeric variables
    is_number = np.vectorize(lambda x: np.issubdtype(x, np.number))
    bool_arr_is_numeric = is_number(data.dtypes)
    vars_numeric = set(data.columns[bool_arr_is_numeric])

    # Remark: numeric variables can still be "categorical"
    # i.e. when they only contain some distinct values!
    # We only consider a variable continuous if they have more distinct values
    # than the requested number bins (using threshold_numeric_is_categorical)

    # continuous if more than threshold_numeric_is_categorical distinct values
    threshold = threshold_numeric_is_categorical
    vars_cat_numeric = set([col for col in vars_numeric
                           if len(data[col].unique()) < threshold])

    # remove from numeric set
    vars_numeric = vars_numeric.difference(vars_cat_numeric)
    # add to categorical set
    vars_cat = vars_cat.union(vars_cat_numeric)

    if id_column_name:
        vars_cat = vars_cat.difference(set([id_column_name]))
    if target_column_name:
        vars_cat = vars_cat.difference(set([target_column_name]))

    return {"numeric_variables": list(vars_numeric),
            "categorical_variables": list(vars_cat)}


def clean_predictor_name(predictor: str) -> str:
    """Strip-off redundant suffix (e.g. "_enc" or "_bin") from the predictor
    name to return a clean version of the predictor

    Args:
        predictor (str): Description

    Returns:
        str: Description
    """
    return (predictor.replace("_enc", "").replace("_bin", "")
            .replace("_processed", ""))
