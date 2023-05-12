
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_squared_error
from numpy import sqrt

import cobra.utils as utils

def compute_univariate_preselection(target_enc_train_data: pd.DataFrame,
                                    target_enc_selection_data: pd.DataFrame,
                                    predictors: list,
                                    target_column: str,
                                    model_type: str = "classification",
                                    preselect_auc_threshold: float = 0.053,
                                    preselect_rmse_threshold: float = 5,
                                    preselect_overtrain_threshold: float = 0.05
                                    ) -> pd.DataFrame:
    """Perform a preselection of predictors based on an AUC (in case of
    classification) or a RMSE (in case of regression) threshold of
    a univariate model on a train and selection dataset and return a DataFrame
    containing for each variable the train and selection AUC or RMSE along with a
    boolean "preselection" column.

    As the AUC just calculates the quality of a ranking, all monotonous
    transformations of a given ranking (i.e. transformations that do not alter
    the ranking itself) will lead to the same AUC.
    Hence, pushing a categorical variable (incl. a binned continuous variable)
    through a logistic regression will produce exactly the same ranking as
    pushing it through incidence replacement (i.e. target encoding),
    as it will produce the exact same output: a ranking of the categories on
    the training set.
    Therefore, no univariate model is trained here as the target encoded train
    and selection data is/must be used as inputs for this function. These will
    be used as predicted scores to compute the AUC with against the target.

    Parameters
    ----------
    model_type : str
        Model type ("classification" or "regression").
    target_enc_train_data : pd.DataFrame
        Train data.
    target_enc_selection_data : pd.DataFrame
        Selection data.
    predictors : list
        List of predictors (e.g. column names in the train set and selection
        data sets).
    target_column : str
        Name of the target column.
    preselect_auc_threshold : float, optional
        Threshold on min. AUC to select predictor. Ignored if model_type is "regression".
    preselect_rmse_threshold : float, optional
        Threshold on max. RMSE to select predictor. Ignored if model_type is "classification".
        It is important to note that the threshold depends heavily on the scale of
        the target variable, and should be modified accordingly.
    preselect_overtrain_threshold : float, optional
        Threshold on the difference between train and selection AUC or RMSE (in case
        of the latter, as a proportion).

    Returns
    -------
    pd.DataFrame
        DataFrame containing for each variable the train AUC or RMSE and
        selection AUC or RMSE along with a boolean indicating whether or not it is
        selected based on the criteria.
    """
    result = []

    if model_type == "classification":
        scoring_method = roc_auc_score
        kwargs = {}
        scoring_method_str = "AUC"
    else:
        scoring_method = mean_squared_error
        kwargs = {"squared": False}
        scoring_method_str = "RMSE"

    for predictor in predictors:
        cleaned_predictor = utils.clean_predictor_name(predictor)

        score_train = scoring_method(
            target_enc_train_data[target_column],
            target_enc_train_data[predictor],
            **kwargs
        )

        score_selection = scoring_method(
            target_enc_selection_data[target_column],
            target_enc_selection_data[predictor],
            **kwargs
        )

        result.append(
            {
                "predictor": cleaned_predictor,
                f"{scoring_method_str} train": score_train,
                f"{scoring_method_str} selection": score_selection
            }
        )

    df_score = pd.DataFrame(result)

    # TODO: This should be `if scoring method is error based` instead of classification vs regression
    if model_type == "classification":
        # Filter based on min. AUC
        score_thresh = df_score.loc[:, f"{scoring_method_str} selection"] > preselect_auc_threshold

        # Identify those variables for which the AUC difference between train
        # and selection is within a user-defined ratio
        score_overtrain = (
            (df_score[f"{scoring_method_str} train"] - df_score[f"{scoring_method_str} selection"])
            < preselect_overtrain_threshold
        )

        df_score["preselection"] = score_thresh & score_overtrain

        df_out = df_score.sort_values(by=f"{scoring_method_str} selection", ascending=False).reset_index(drop=True)
    else:
        # What if they fill in something else than `regression`?
        # Filter based on max. RMSE
        score_thresh = df_score.loc[:, f"{scoring_method_str} selection"] < preselect_rmse_threshold

        # Identify those variables for which the RMSE difference between train
        # and selection is within a user-defined ratio
        score_overtrain = (
            (df_score[f"{scoring_method_str} selection"] - df_score[f"{scoring_method_str} train"])  # flip subtraction vs. AUC
            < preselect_overtrain_threshold
        )

        df_score["preselection"] = score_thresh & score_overtrain

        df_out = df_score.sort_values(by=f"{scoring_method_str} selection", ascending=True).reset_index(drop=True)  # lower is better
    return df_out

def get_preselected_predictors(df_metric: pd.DataFrame) -> list:
    """Wrapper function to extract a list of predictors from df_metric.

    Parameters
    ----------
    df_metric : pd.DataFrame
        DataFrame containing for each variable the train AUC or RMSE and
        test AUC or RMSE along with a boolean indicating whether or not it is selected
        based on the criteria.

    Returns
    -------
    list
        List of preselected predictors.
    """

    if "AUC selection" in df_metric.columns:
        predictor_list = (df_metric[df_metric["preselection"]]
                          .sort_values(by="AUC selection", ascending=False)
                          .predictor.tolist())
    elif "RMSE selection" in df_metric.columns:
        predictor_list = (df_metric[df_metric["preselection"]]
                          .sort_values(by="RMSE selection", ascending=True)  # lower is better
                          .predictor.tolist())

    return [col + "_enc" for col in predictor_list]

def compute_correlations(target_enc_train_data: pd.DataFrame,
                         predictors: list) -> pd.DataFrame:
    """Given a DataFrame and a list of predictors, compute the correlations
    amongst the predictors in the DataFrame.

    Parameters
    ----------
    target_enc_train_data : pd.DataFrame
        Data to compute correlation.
    predictors : list
        List of column names of the DataFrame between which to compute
        the correlation matrix.

    Returns
    -------
    pd.DataFrame
        The correlation matrix of the training set.
    """

    correlations = target_enc_train_data[predictors].corr()

    predictors_cleaned = [utils.clean_predictor_name(predictor)
                          for predictor in predictors]

    # Change index and columns with the cleaned version of the predictors
    # e.g. change "var1_enc" with "var1"
    correlations.columns = predictors_cleaned
    correlations.index = predictors_cleaned

    return correlations
