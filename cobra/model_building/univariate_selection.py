"""
Module to perform univariate preselection and compute correlation amongst
predictors
Authors:
- Geert Verstraeten (methodology)
- Matthias Roels (current implementation)
- Jan Benisek (initial implementation)
"""
import pandas as pd
from sklearn.metrics import roc_auc_score
import cobra.utils as utils


def compute_univariate_preselection(target_enc_train_data: pd.DataFrame,
                                    target_enc_selection_data: pd.DataFrame,
                                    predictors: list,
                                    target_column: str,
                                    preselect_auc_threshold: float=0.053,
                                    preselect_overtrain_threshold: float=0.05
                                    ) -> pd.DataFrame:
    """Perform a preselection of predictors based on an AUC threshold of
    a univariate model on a train and selection dataset and return a datframe
    containing for each variable the train and selection AUC along with a
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
    be used as predicted scores to compute the AUC with against the target

    Parameters
    ----------
    target_enc_train_data : pd.DataFrame
        Train data
    target_enc_selection_data : pd.DataFrame
        Selection data
    predictors : list
        list of predictors (e.g. column names in the train set and selection
        data sets)
    target_column : str
        name of the target column
    preselect_auc_threshold : float, optional
        threshold on AUC to select predictor
    preselect_overtrain_threshold : float, optional
        threshold on the difference between train and selection AUC

    Returns
    -------
    pd.DataFrame
        DataFrame containing for each variable the train auc and
        selection auc allong with a boolean indicating whether or not it is
        selected based on the criteria
    """
    result = []

    for predictor in predictors:

        cleaned_predictor = utils.clean_predictor_name(predictor)

        auc_train = roc_auc_score(
            y_true=target_enc_train_data[target_column],
            y_score=target_enc_train_data[predictor])

        auc_selection = roc_auc_score(
            y_true=target_enc_selection_data[target_column],
            y_score=target_enc_selection_data[predictor]
            )

        result.append({"predictor": cleaned_predictor,
                       "AUC train": auc_train,
                       "AUC selection": auc_selection})

    df_auc = pd.DataFrame(result)

    # Filter based on min AUC
    auc_thresh = df_auc.loc[:, "AUC selection"] > preselect_auc_threshold

    # Identify those variables for which the AUC difference between train
    # and selection is within a user-defined ratio
    auc_overtrain = ((df_auc["AUC train"] - df_auc["AUC selection"])
                     < preselect_overtrain_threshold)

    df_auc["preselection"] = auc_thresh & auc_overtrain

    return (df_auc.sort_values(by='AUC selection', ascending=False)
            .reset_index(drop=True))


def get_preselected_predictors(df_auc: pd.DataFrame) -> list:
    """Wrapper function to extract a list of predictors from df_auc

    Parameters
    ----------
    df_auc : pd.DataFrame
        DataFrame containing for each variable the train auc and
        test auc allong with a boolean indicating whether or not it is selected
        based on the criteria
    Returns
    -------
    list
        list of preselected predictors
    """
    predictor_list = (df_auc[df_auc["preselection"]]
                      .sort_values(by='AUC selection', ascending=False)
                      .predictor.tolist())

    return [col + "_enc" for col in predictor_list]


def compute_correlations(target_enc_train_data: pd.DataFrame,
                         predictors: list) -> pd.DataFrame:
    """Given a DataFrame and a list of predictors, compute the correlations
    amongst the predictors in the DataFrame

    Parameters
    ----------
    target_enc_train_data : pd.DataFrame
        data to compute correlation
    predictors : list
        List of column names of the DataFrame between which to compute
        the correlation matrix

    Returns
    -------
    pd.DataFrame
        The correlation matrix of the training set
    """

    correlations = target_enc_train_data[predictors].corr()

    predictors_cleaned = [utils.clean_predictor_name(predictor)
                          for predictor in predictors]

    # Change index and columns with the cleaned version of the predictors
    # e.g. change "var1_enc" with "var1"
    correlations.columns = predictors_cleaned
    correlations.index = predictors_cleaned

    return correlations
