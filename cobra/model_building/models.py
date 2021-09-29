# third party imports
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, mean_squared_error
from numpy import sqrt
from sklearn.linear_model import LogisticRegression, LinearRegression
# custom imports
import cobra.utils as utils


class LogisticRegressionModel:
    """Wrapper around the LogisticRegression class, with additional methods
    implemented such as evaluation (using AUC), getting a list of coefficients,
    a dictionary of coefficients per predictor, ... for convenience

    Attributes
    ----------
    logit : LogisticRegression
        scikit-learn logistic regression model.
    predictors : list
        List of predictors used in the model.
    """

    def __init__(self):
        self.logit = LogisticRegression(fit_intercept=True, C=1e9,
                                        solver='liblinear', random_state=42)
        self._is_fitted = False
        # placeholder to keep track of a list of predictors
        self.predictors = []
        self._eval_metrics_by_split = {}

    def serialize(self) -> dict:
        """Serialize model as JSON

        Returns
        -------
        dict
            dictionary containing the serialized JSON
        """
        serialized_model = {
            "meta": "logistic-regression",
            "predictors": self.predictors,
            "_eval_metrics_by_split": self._eval_metrics_by_split,
            "params": self.logit.get_params()
        }

        if self._is_fitted:
            serialized_model.update({
                "classes_": self.logit.classes_.tolist(),
                "coef_": self.logit.coef_.tolist(),
                "intercept_": self.logit.intercept_.tolist(),
                "n_iter_": self.logit.n_iter_.tolist(),
            })

        return serialized_model

    def deserialize(self, model_dict: dict):
        """Deserialize a model previously stored as JSON

        Parameters
        ----------
        model_dict : dict
            Serialized JSON file as a dict.

        Raises
        ------
        ValueError
            In case JSON file is no valid serialized model
        """

        if not self._is_valid_dict(model_dict):
            raise ValueError("No valid serialized model")

        self.logit = LogisticRegression()
        self.logit.set_params(**model_dict["params"])
        self.logit.classes_ = np.array(model_dict["classes_"])
        self.logit.coef_ = np.array(model_dict["coef_"])
        self.logit.intercept_ = np.array(model_dict["intercept_"])
        self.logit.n_iter_ = np.array(model_dict["intercept_"])
        self.predictors = model_dict["predictors"]
        self._eval_metrics_by_split = model_dict["_eval_metrics_by_split"]

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
            Predictors of train data.
        y_train : pd.Series
            Target of train data.
        """
        self.predictors = list(X_train.columns)
        self.logit.fit(X_train, y_train)
        self._is_fitted = True

    def score_model(self, X: pd.DataFrame) -> np.ndarray:
        """Score a model on a (new) dataset

        Parameters
        ----------
        X : pd.DataFrame
            Dataset of predictors to score the model.

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
            Dataset containing the predictor values for each observation.
        y : pd.Series
            Dataset containing the target of each observation.
        split : str, optional
            Split of the dataset (e.g. train-selection-validation).

        Returns
        -------
        float
            the performance score of the model (AUC)
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
            Data to score the model.

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
                                    orient="index").reset_index()
        df.columns = ["predictor", "importance"]

        return (df.sort_values(by="importance", ascending=False)
                .reset_index(drop=True))

    def _is_valid_dict(self, model_dict: dict) -> bool:

        if ("meta" not in model_dict
                or model_dict["meta"] != "logistic-regression"):
            return False

        attr = ["classes_", "coef_", "intercept_", "n_iter_", "predictors"]
        for key in attr:
            if not (key in model_dict or type(model_dict[key]) != list):
                return False

        if ("params" not in model_dict
                or "_eval_metrics_by_split" not in model_dict):
            return False

        return True


class LinearRegressionModel:
    """Wrapper around the LinearRegression class, with additional methods
    implemented such as evaluation (using RMSE), getting a list of coefficients,
    a dictionary of coefficients per predictor, ... for convenience

    Attributes
    ----------
    linear : LinearRegression
        scikit-learn linear regression model.
    predictors : list
        List of predictors used in the model.
    """

    def __init__(self):
        self.linear = LinearRegression(fit_intercept=True, normalize=False)
        self._is_fitted = False
        # placeholder to keep track of a list of predictors
        self.predictors = []
        self._eval_metrics_by_split = {}

    def serialize(self) -> dict:
        """Serialize model as JSON

        Returns
        -------
        dict
            dictionary containing the serialized JSON
        """
        serialized_model = {
            "meta": "linear-regression",
            "predictors": self.predictors,
            "_eval_metrics_by_split": self._eval_metrics_by_split,
            "params": self.linear.get_params()
        }

        if self._is_fitted:
            serialized_model.update({
                "coef_": self.linear.coef_.tolist(),
                "intercept_": self.linear.intercept_.tolist()
            })

        return serialized_model

    def deserialize(self, model_dict: dict):
        """Deserialize a model previously stored as JSON

        Parameters
        ----------
        model_dict : dict
            Serialized JSON file as a dict.

        Raises
        ------
        ValueError
            In case JSON file is no valid serialized model
        """

        if not self._is_valid_dict(model_dict):
            raise ValueError("No valid serialized model")

        self.linear = LinearRegression()
        self.linear.set_params(**model_dict["params"])
        self.linear.coef_ = np.array(model_dict["coef_"])
        self.linear.intercept_ = np.array(model_dict["intercept_"])
        self.predictors = model_dict["predictors"]
        self._eval_metrics_by_split = model_dict["_eval_metrics_by_split"]

    def get_coef(self) -> np.array:
        """Returns the model coefficients

        Returns
        -------
        np.array
            array of model coefficients
        """
        return self.linear.coef_[0]

    def get_intercept(self) -> float:
        """Returns the intercept of the model

        Returns
        -------
        float
            intercept of the model
        """
        return self.linear.intercept_[0]

    def get_coef_by_predictor(self) -> dict:
        """Returns a dictionary mapping predictor (key) to coefficient (value)

        Returns
        -------
        dict
            map ``{predictor: coefficient}``
        """
        return dict(zip(self.predictors, self.linear.coef_[0]))

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Fit the model

        Parameters
        ----------
        X_train : pd.DataFrame
            Predictors of train data.
        y_train : pd.Series
            Target of train data.
        """
        self.predictors = list(X_train.columns)
        self.linear.fit(X_train, y_train)
        self._is_fitted = True

    def score_model(self, X: pd.DataFrame) -> np.ndarray:
        """Score a model on a (new) dataset

        Parameters
        ----------
        X : pd.DataFrame
            Dataset of predictors to score the model.

        Returns
        -------
        np.ndarray
            score of the model for each observation
        """
        # We select predictor columns (self.predictors) here to
        # ensure we have the proper predictors and the proper order!!!
        return self.linear.predict(X[self.predictors])

    def evaluate(self, X: pd.DataFrame, y: pd.Series,
                 split: str=None) -> float:
        """Evaluate the model on a given data set (X, y). The optional split
        parameter is to indicate that the data set belongs to
        (train, selection, validation), so that the computation on these sets
        can be cached!

        Parameters
        ----------
        X : pd.DataFrame
            Dataset containing the predictor values for each observation.
        y : pd.Series
            Dataset containing the target of each observation.
        split : str, optional
            Split of the dataset (e.g. train-selection-validation).

        Returns
        -------
        float
            the performance score of the model (RMSE)
        """

        if (split is None) or (split not in self._eval_metrics_by_split):

            y_pred = self.score_model(X)

            performance = sqrt(mean_squared_error(y_true=y, y_pred=y_pred))

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
            Data to score the model.

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
                                    orient="index").reset_index()
        df.columns = ["predictor", "importance"]

        return (df.sort_values(by="importance", ascending=False)
                .reset_index(drop=True))

    def _is_valid_dict(self, model_dict: dict) -> bool:

        if ("meta" not in model_dict
                or model_dict["meta"] != "linear-regression"):
            return False

        attr = ["coef_", "intercept_", "predictors"]
        for key in attr:
            if not (key in model_dict or type(model_dict[key]) != list):
                return False

        if ("params" not in model_dict
                or "_eval_metrics_by_split" not in model_dict):
            return False

        return True
