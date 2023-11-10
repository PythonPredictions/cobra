import inspect
from abc import abstractmethod
from typing import Callable, Optional

# third party imports
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, mean_squared_error
from numpy import sqrt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_curve

# custom imports
import cobra.utils as utils
from cobra.evaluation import ClassificationEvaluator


class Model:
    """Base class for all models provided in Cobra.

    Attributes
    ----------
    linear : LinearRegression
        scikit-learn linear regression model.
    predictors : list
        List of predictors used in the model.
    """

    def __init__(self):
        self._is_fitted = False
        self.predictors = []
        # Cache for model performance per dataset split:
        self._performance_per_split = {}

    def serialize(self) -> dict:
        """Serialize model as JSON.

        Returns
        -------
        dict
            Dictionary containing the serialized JSON.
        """
        serialized_model = {
            "predictors": self.predictors,
            "_performance_per_split": self._performance_per_split,
        }
        return serialized_model

    def deserialize(self, model_dict: dict):
        """Deserialize a model previously stored as JSON.

        Parameters
        ----------
        model_dict : dict
            Serialized JSON file as a dict.

        Raises
        ------
        ValueError
            In case JSON file is no valid serialized model.
        """

        if not self._is_valid_dict(model_dict):
            raise ValueError("No valid serialized model")

        self.predictors = model_dict["predictors"]

        if "_performance_per_split" in model_dict.keys():
            self._performance_per_split = model_dict["_performance_per_split"]
        elif "_eval_metrics_by_split" in model_dict.keys():  # backwards compat.
            self._performance_per_split = model_dict["_eval_metrics_by_split"]

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Fit the model.

        Any child class should call _after_fit() at the end.

        Parameters
        ----------
        X_train : pd.DataFrame
            Predictors of train data.
        y_train : pd.Series
            Target of train data.
        """
        pass

    @abstractmethod
    def score_model(self, X: pd.DataFrame) -> np.ndarray:
        """Score a model on a (new) dataset.

        Parameters
        ----------
        X : pd.DataFrame
            Dataset of predictors to score the model.

        Returns
        -------
        np.ndarray
            Score (i.e. predicted probabilities) of the model for each observation.
        """
        pass

    def evaluate(self, X: pd.DataFrame, y: pd.Series,
                 split: str = None,
                 metric: Optional[Callable] = None,
                 metric_args: Optional[dict] = None,
                 metric_kwargs: Optional[dict] = None) -> float:
        """Evaluate the model on a given dataset (X, y).

        Caching of the calculated model performance can be enabled with the
        split argument for faster execution, and a custom metric can be
        configured for the calculation of the model performance.

        Parameters
        ----------
        X : pd.DataFrame
            Dataset containing the predictor values for each observation.
        y : pd.Series
            Dataset containing the target of each observation.
        split : str, optional
            Name of the split of the dataset ("train", "selection",
            or "validation") for which the model performance must be calculated.
            Providing this argument will enable caching the calculated model
            performance for the given dataset split, such that every time the
            performance for the same split is requested again,
            the performance must not be re-calculated.
        metric : Callable (function), optional
            Function that evaluates the model's performance, by calculating a
            certain evaluation metric.
            If the metric is not provided, the default metric RMSE is used for
            evaluating the model for a regression task, or AUC for a
            classification task.
            The metric functions from sklearn can be used, see
            https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics.
            You can also pass a custom function.
            The variables that we provide and your function can possibly take in
            are y_true, y_pred, y_score and y_prob arguments, of numpy array
            type, and optionally also additional arguments, which you can pass
            through the metric_args and metric_kwargs parameters.
            If you are unsure which arguments of your metric function are
            args/kwargs, then run inspect.getfullargspec(your_metric_function).

            Example functions for regression:
            - sklearn.metrics.r2_score(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average')
            - overall_estimated_commission_earned(y_true, y_pred,
                                                  avg_prob_buy_if_err_lower_than_20K=0.25,
                                                  avg_prob_buy_if_err_higher_than_20K=0.05,
                                                  pct_commission_on_buy=0.05)

            Example functions for classification:
            - ClassificationEvaluator._compute_lift(y_true=y_true,
                                                    y_score=y_pred,
                                                    lift_at=0.05)
            - return_on_investment(y_true, y_pred,
                                  cost_of_letter=2.10,
                                  success_rate_of_letter=0.25,
                                  average_return_for_successful_letter=75.0)

        metric_args : dict, optional
            Arguments to be passed to the metric function when
            evaluating the model's performance.
            See the documentation of the metric function above.
            Example arguments are: lift_at for
            ClassificationEvaluator._compute_lift()
            or pct_commission_on_buy for overall_estimated_commission_earned().
        metric_kwargs : dict, optional
            Keyword arguments (for example: normalize=True) to be passed to the
            metric function when evaluating the model's performance.
            See the documentation of the metric function above.
            An example keyword argument is sample_weight for
            sklearn.metrics.r2_score().

        Returns
        -------
        float
            The performance score of the model (RMSE by default for
            regression, AUC by default for classification).
        """
        # If the performance is requested for a certain split of the dataset
        # and it has been pre-calculated already (it is available in the cache),
        # return it immediately from the cache:
        if split is not None and split in self._performance_per_split:
            return self._performance_per_split[split]

        y_score = self.score_model(X)

        if metric is None:
            # No custom evaluation metric was chosen.
            performance = self._evaluate_with_default_metric(y, y_score)

        else:
            # A custom evaluation metric was chosen.
            # Compute the model performance with the chosen metric function,
            # pass all arguments this function could potentially need,
            # including optional keyword arguments that were passed when
            # initializing this model.
            args = self._prepare_args_for_custom_evaluation_metric(y,
                                                                   y_score,
                                                                   metric)
            if metric_args is not None and isinstance(metric_args, dict):
                args = {**args, **metric_args}
            args = {
                arg: val
                for arg, val in args.items()
                # we can't provide too many arguments vs. the args of the
                # metric's signature:
                if arg in inspect.getfullargspec(metric).args
            }
            if metric_kwargs is None:
                metric_kwargs = {}
            performance = metric(**args, **metric_kwargs)

        if split is None:
            return performance
        else:
            # Cache the model performance for the given dataset split:
            self._performance_per_split[split] = performance
            return self._performance_per_split[split]

    def compute_variable_importance(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute the importance of each predictor in the model and return
        it as a DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            Data to score the model.

        Returns
        -------
        pd.DataFrame
            DataFrame containing columns predictor and importance.
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

    def _is_valid_dict(self, model_dict: dict):
        """Verify whether a model dictionary, previously stored as JSON,
        contains valid information, before constructing a model class from it.

        Parameters
        ----------
        model_dict : dict
            Serialized JSON file as a dict.

        Raises
        ------
        ValueError
            In case JSON file is no valid serialized model.
        """
        for key in ["predictors", "params"]:
            if key not in model_dict:
                return False

        if "_performance_per_split" not in model_dict \
                and "_eval_metrics_by_split" not in model_dict:  # backw compat.
            return False

        return True

    @abstractmethod
    def _evaluate_with_default_metric(self, y_true, y_score):
        pass

    def _after_fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Fit the model.

        Parameters
        ----------
        X_train : pd.DataFrame
            Predictors of train data.
        y_train : pd.Series
            Target of train data.
        """
        self.predictors = list(X_train.columns)
        self._is_fitted = True

    @abstractmethod
    def _prepare_args_for_custom_evaluation_metric(self, y_true, y_score,
                                                   metric):
        pass


class LogisticRegressionModel(Model):
    """Wrapper around sklearn's LogisticRegression class, with additional
    methods implemented such as evaluation (using AUC), getting a list of
    coefficients, a dictionary of coefficients per predictor, ... for
    convenience.

    See also the documentation of the base class (Model).

    Attributes
    ----------
    logit : LogisticRegression
        The scikit-learn logistic regression model that is trained and
        afterwards used for making predictions.
    """

    def __init__(self):
        super().__init__()
        self.logit = LogisticRegression(fit_intercept=True, C=1e9,
                                        solver='liblinear', random_state=42)

    def serialize(self) -> dict:
        """Serialize model as JSON.

        Returns
        -------
        dict
            Dictionary containing the serialized JSON.
        """
        serialized_model = super().serialize()
        serialized_model["meta"] = "logistic-regression"
        serialized_model["params"] = self.logit.get_params()

        if self._is_fitted:
            serialized_model.update({
                "classes_": self.logit.classes_.tolist(),
                "coef_": self.logit.coef_.tolist(),
                "intercept_": self.logit.intercept_.tolist(),
                "n_iter_": self.logit.n_iter_.tolist(),
            })

        return serialized_model

    def deserialize(self, model_dict: dict):
        """Deserialize a model previously stored as JSON.

        Parameters
        ----------
        model_dict : dict
            Serialized JSON file as a dict.

        Raises
        ------
        ValueError
            In case JSON file is no valid serialized model.
        """
        super().deserialize(model_dict)
        self.logit = LogisticRegression()
        self.logit.set_params(**model_dict["params"])
        self.logit.classes_ = np.array(model_dict["classes_"])
        self.logit.coef_ = np.array(model_dict["coef_"])
        self.logit.intercept_ = np.array(model_dict["intercept_"])
        self.logit.n_iter_ = np.array(model_dict["intercept_"])

    def get_coef(self) -> np.array:
        """Returns the model coefficients.

        Returns
        -------
        np.array
            Array of model coefficients.
        """
        return self.logit.coef_[0]

    def get_intercept(self) -> float:
        """Returns the intercept of the model.

        Returns
        -------
        float
            Intercept of the model.
        """
        return self.logit.intercept_[0]

    def get_coef_by_predictor(self) -> dict:
        """Returns a dictionary mapping predictor (key) to coefficient (value).

        Returns
        -------
        dict
            A map ``{predictor: coefficient}``.
        """
        return dict(zip(self.predictors, self.logit.coef_[0]))

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Fit the model.

        Parameters
        ----------
        X_train : pd.DataFrame
            Predictors of train data.
        y_train : pd.Series
            Target of train data.
        """
        self.logit.fit(X_train, y_train)
        super()._after_fit(X_train, y_train)

    def score_model(self, X: pd.DataFrame) -> np.ndarray:
        """Score a model on a (new) dataset.

        Parameters
        ----------
        X : pd.DataFrame
            Dataset of predictors to score the model.

        Returns
        -------
        np.ndarray
            Score (i.e. predicted probabilities) of the model for each observation.
        """
        # We select predictor columns (self.predictors) here to
        # ensure we have the proper predictors and the proper order
        return self.logit.predict_proba(X[self.predictors])[:, 1]

    def _is_valid_dict(self, model_dict: dict) -> bool:
        if not super()._is_valid_dict(model_dict):
            return False

        if ("meta" not in model_dict
                or model_dict["meta"] != "logistic-regression"):
            return False

        attr = ["classes_", "coef_", "intercept_", "n_iter_"]
        for key in attr:
            if not (key in model_dict or type(model_dict[key]) != list):
                return False

        return True

    def _evaluate_with_default_metric(self, y_true, y_score):
        """We use AUC as default evaluation metric."""
        return roc_auc_score(y_true=y_true, y_score=y_score)

    def _prepare_args_for_custom_evaluation_metric(self, y_true, y_score,
                                                   metric):
        args = {
            "y_true": y_true,
            "y_score": y_score,
            "y_prob": y_score
        }

        if "y_pred" in inspect.getfullargspec(metric).args:
            # With the default metric AUC, the performance could be
            # scored over all possible thresholds and based on y_score;
            # now, with any evaluation metric possibly being used, y_pred
            # may be required instead of y_score, which requires determining
            # the optimal threshold first and then calculating y_pred.
            fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score)
            cutoff = ClassificationEvaluator._compute_optimal_cutoff(fpr,
                                                                     tpr,
                                                                     thresholds)
            args["y_pred"] = np.array([0 if score <= cutoff
                                       else 1
                                       for score in y_score])
        return args


class LinearRegressionModel(Model):
    """Wrapper around sckit-learn's LinearRegression class,
    with additional methods implemented such as evaluation (using RMSE),
    getting a list of coefficients, a dictionary of coefficients per
    predictor,... for convenience.

    See also the documentation of the base class (Model).

    Attributes
    ----------
    linear : LinearRegression
        scikit-learn linear regression model.
    """

    def __init__(self):
        super().__init__()
        self.linear = LinearRegression(fit_intercept=True)

    def serialize(self) -> dict:
        """Serialize model as JSON.

        Returns
        -------
        dict
            Dictionary containing the serialized JSON.
        """
        serialized_model = super().serialize()
        serialized_model["meta"] = "linear-regression"
        serialized_model["params"] = self.linear.get_params()

        if self._is_fitted:
            serialized_model.update({
                "coef_": self.linear.coef_.tolist(),
                "intercept_": self.linear.intercept_.tolist()
            })

        return serialized_model

    def deserialize(self, model_dict: dict):
        """Deserialize a model previously stored as JSON.

        Parameters
        ----------
        model_dict : dict
            Serialized JSON file as a dict.

        Raises
        ------
        ValueError
            In case JSON file is no valid serialized model.
        """
        super().deserialize(model_dict)
        self.linear = LinearRegression()
        self.linear.set_params(**model_dict["params"])
        self.linear.coef_ = np.array(model_dict["coef_"])
        self.linear.intercept_ = np.array(model_dict["intercept_"])

    def get_coef(self) -> np.array:
        """Returns the model coefficients.

        Returns
        -------
        np.array
            Array of model coefficients.
        """
        return self.linear.coef_

    def get_intercept(self) -> float:
        """Returns the intercept of the model.

        Returns
        -------
        float
            Intercept of the model.
        """
        return self.linear.intercept_[0]

    def get_coef_by_predictor(self) -> dict:
        """Returns a dictionary mapping predictor (key) to coefficient (value).

        Returns
        -------
        dict
            A map ``{predictor: coefficient}``.
        """
        return dict(zip(self.predictors, self.linear.coef_))

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Fit the model.

        Parameters
        ----------
        X_train : pd.DataFrame
            Predictors of train data.
        y_train : pd.Series
            Target of train data.
        """
        self.linear.fit(X_train, y_train)
        super()._after_fit(X_train, y_train)

    def score_model(self, X: pd.DataFrame) -> np.ndarray:
        """Score a model on a (new) dataset.

        Parameters
        ----------
        X : pd.DataFrame
            Dataset of predictors to score the model.

        Returns
        -------
        np.ndarray
            Score of the model for each observation.
        """
        # We select predictor columns (self.predictors) here to
        # ensure we have the proper predictors and the proper order
        return self.linear.predict(X[self.predictors])

    def _is_valid_dict(self, model_dict: dict) -> bool:

        if not super()._is_valid_dict(model_dict):
            return False

        if ("meta" not in model_dict
                or model_dict["meta"] != "linear-regression"):
            return False

        attr = ["coef_", "intercept_"]
        for key in attr:
            if not (key in model_dict or type(model_dict[key]) != list):
                return False

        return True

    def _evaluate_with_default_metric(self, y_true, y_score):
        """We use RMSE as default evaluation metric."""
        return sqrt(
            mean_squared_error(y_true=y_true,
                               # LinearRegressionModel actually returns y_pred
                               # inside y_score:
                               y_pred=y_score)
        )

    def _prepare_args_for_custom_evaluation_metric(self, y_true, y_score,
                                                   metric):
        return {
            "y_true": y_true,
            # LinearRegressionModel actually returns y_pred inside y_score:
            "y_pred": y_score
        }
