# third party imports
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
# custom imports
import cobra.utils as utils


class LogisticRegressionModel:

    """Wrapper around the LogisticRegression class, with additional methods
    implemented such as evaluation (using auc), getting a list of coefficients,
    a ditionary of coefficients per predictor, ... for convenience

    Attributes
    ----------
    logit : LogisticRegression
        scikit-learn logistic regression model
    predictors : list
        List of predictors used in the model
    """

    def __init__(self):
        self.logit = LogisticRegression(fit_intercept=True, C=1e9,
                                        solver='liblinear', random_state=42)
        # placeholder to keep track of a list of predictors
        self.predictors = []
        self._eval_metrics_by_split = {}

    def get_coef(self) -> np.array:
        """Returns the model coefficients

        Returns
        -------
        np.array
            array of model coefficients
        """
        return self.logit.coef_[0]

    def get_intercept(self) -> float:
        """Returns the intercept of the model

        Returns
        -------
        float
            intercept of the model
        """
        return self.logit.intercept_[0]

    def get_coef_by_predictor(self) -> dict:
        """Returns a dictionary mapping predictor (key) to coefficient (value)

        Returns
        -------
        dict
            map ``{predictor: coefficient}``
        """
        return dict(zip(self.predictors, self.logit.coef_[0]))

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Fit the model

        Parameters
        ----------
        X_train : pd.DataFrame
            predictors of train data
        y_train : pd.Series
            target of train data
        """
        self.predictors = list(X_train.columns)
        self.logit.fit(X_train, y_train)

    def score_model(self, X: pd.DataFrame) -> np.ndarray:
        """Score a model on a (new) dataset

        Parameters
        ----------
        X : pd.DataFrame
            dataset of predictors to score the model

        Returns
        -------
        np.ndarray
            score of the model for each observation
        """
        # We select predictor columns (self.predictors) here to
        # ensure we have the proper predictors and the proper order!!!
        return self.logit.predict_proba(X[self.predictors])[:, 1]

    def evaluate(self, X: pd.DataFrame, y: pd.Series,
                 split: str=None) -> float:
        """Evaluate the model on a given data set (X, y). The optional split
        parameter is to indicate that the data set belongs to
        (train, selection, validation), so that the computation on these sets
        can be cached!

        Parameters
        ----------
        X : pd.DataFrame
            dataset containing the predictor values for each observation
        y : pd.Series
            dataset containig the target of each observation
        split : str, optional
            split of the dataset (e.g. train-selection-validation)

        Returns
        -------
        float
            the performance score of the model (e.g. AUC)
        """

        if (split is None) or (split not in self._eval_metrics_by_split):

            y_pred = self.score_model(X)

            performance = roc_auc_score(y_true=y, y_score=y_pred)

            if split is None:
                return performance
            else:
                self._eval_metrics_by_split[split] = performance

        return self._eval_metrics_by_split[split]

    def compute_variable_importance(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute the importance of each predictor in the model and return
        it as a DataFrame

        Parameters
        ----------
        data : pd.DataFrame
            data to score the model

        Returns
        -------
        pd.DataFrame
            DataFrame containing columns predictor and importance
        """

        y_pred = self.score_model(data)

        importance_by_variable = {
            utils.clean_predictor_name(predictor): stats.pearsonr(
                data[predictor],
                y_pred
                )[0]
            for predictor in self.predictors
        }

        df = pd.DataFrame.from_dict(importance_by_variable,
                                    orient='index').reset_index()
        df.columns = ["predictor", "importance"]

        return (df.sort_values(by="importance", ascending=False)
                .reset_index(drop=True))
