import numpy as np
import pandas as pd


def get_column_datatypes(data: pd.DataFrame,
                         target_column_name: str=None,
                         id_column_name: str=None,
                         numeric_is_categorical_threshold: int=10) -> dict:
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
    numeric_is_categorical_threshold : int, optional
        Threshold to decide whether a numeric variable is categorical based
        on the number of unique values in that column

    Returns
    -------
    dict
        Description
    """
    column_names = list(data.columns)

    # dummies variables: case they have only 2 values
    vars_dummy = set([col for col in column_names
                      if len(data[col].unique()) == 2])

    # categorical vars
    vars_cat = (set(data.dtypes[data.dtypes == object].index)
                .union(set(data.dtypes[data.dtypes == "category"].index)))

    # Numeric variables
    is_number = np.vectorize(lambda x: np.issubdtype(x, np.number))
    bool_arr_is_numeric = is_number(data.dtypes)
    vars_numeric = set(data.columns[bool_arr_is_numeric])

    # remove dummy variables from set
    vars_numeric = vars_numeric.difference(vars_dummy)

    # Remark: numeric variables can still be "categorical"
    # i.e. when they only contain some distinct values!
    # We only consider a variable continuous if they have more distinct values
    # than the requested number bins (using numeric_is_categorical_threshold)

    # continuous if more than numeric_is_categorical_threshold distinct values
    threshold = numeric_is_categorical_threshold
    vars_cat_numeric = set([col for col in vars_numeric
                           if len(data[col].unique()) < threshold])

    # remove from numeric set
    vars_numeric = vars_numeric.difference(vars_cat_numeric)
    # add to categorical set
    vars_cat = vars_cat.union(vars_cat_numeric)

    if id_column_name:
        vars_cat = vars_cat.difference(set([id_column_name]))
    if target_column_name:
        vars_dummy = vars_dummy.difference(set([target_column_name]))

    return {"numeric_variables": vars_numeric,
            "categorical_variables": vars_cat,
            "dummy_variables": vars_dummy}
