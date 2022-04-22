"""Evaluate the created model."""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from numpy import sqrt
from scipy.stats import norm

# classification
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.exceptions import NotFittedError

# regression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


DEFAULT_LABELS = ["0", "1"]


class ClassificationEvaluator():
    """Evaluator class encapsulating classification model metrics and plotting functionality.

    Attributes
    ----------
    y_true : np.ndarray
        True binary target data labels.
    y_pred : np.ndarray
        Target scores of the model.
    confusion_matrix : np.ndarray
        Confusion matrix computed for a particular cut-off.
    cumulative_gains : tuple
        Data for plotting cumulative gains curve.
    evaluation_metrics : dict
        Map containing various scalar evaluation metrics (precision, recall, accuracy, AUC, F1, etc.).
    lift_at : float
        Parameter to determine at which top level percentage the lift of the
        model should be computed.
    lift_curve : tuple
        Data for plotting lift curve(s).
    probability_cutoff : float
        Probability cut off to convert probability scores to a binary score.
    roc_curve : dict
        Map containing true-positive-rate, false-positive-rate at various
        thresholds (also incl.).
    n_bins : int, optional
        Defines the number of bins used to calculate the lift curve for
        (by default 10, so deciles).
    """

    def __init__(
        self,
        probability_cutoff: float = None,
        lift_at: float = 0.05,
        n_bins: int = 10
    ):
        """Initialize the ClassificationEvaluator."""
        self.y_true = None
        self.y_pred = None

        self.lift_at = lift_at
        self.probability_cutoff = probability_cutoff
        self.n_bins = n_bins

        # Placeholder to store fitted output
        self.scalar_metrics = None
        self.roc_curve = None
        self.confusion_matrix = None
        self.lift_curve = None
        self.cumulative_gains = None

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Fit the evaluator by computing the relevant evaluation metrics on the inputs.

        Parameters
        ----------
        y_true : np.ndarray
            True labels.
        y_pred : np.ndarray
            Model scores (as probability).
        """
        fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_pred)

        # if probability_cutoff is not set, take the optimal cut-off
        if not self.probability_cutoff:
            self.probability_cutoff = (ClassificationEvaluator.
                                       _compute_optimal_cutoff(fpr, tpr,
                                                               thresholds))

        # Transform probabilities to binary array using cut-off
        y_pred_b = np.array([0 if pred <= self.probability_cutoff else 1
                             for pred in y_pred])

        # Compute the various evaluation metrics
        self.scalar_metrics = ClassificationEvaluator._compute_scalar_metrics(
            y_true,
            y_pred,
            y_pred_b,
            self.lift_at
        )

        self.y_true = y_true
        self.y_pred = y_pred

        self.roc_curve = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}
        self.confusion_matrix = confusion_matrix(y_true, y_pred_b)
        self.lift_curve = ClassificationEvaluator._compute_lift_per_bin(y_true, y_pred, self.n_bins)
        self.cumulative_gains = ClassificationEvaluator._compute_cumulative_gains(y_true, y_pred)

    @staticmethod
    def _compute_scalar_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_b: np.ndarray,
        lift_at: float
    ) -> pd.Series:
        """Compute various scalar performance measures.

        Parameters
        ----------
        y_true : np.ndarray
            True binary target data labels.
        y_pred : np.ndarray
            Target scores of the model.
        y_pred_b : np.ndarray
            Predicted target data labels (binary).
        lift_at : float
            At what top level percentage the lift should be computed.

        Returns
        -------
        pd.Series
            Contains various performance measures of the model, being:
                Accuracy
                AUC
                Precision
                Recall
                F1
                Matthews correlation coefficient
                Lift at given percentage

        Raises
        ----------
        ValueError
            The `column_order` and `pig_tables` parameters do not contain
            the same set of variables.
        """
        return pd.Series({
            "accuracy": accuracy_score(y_true, y_pred_b),
            "AUC": roc_auc_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred_b),
            "recall": recall_score(y_true, y_pred_b),
            "F1": f1_score(y_true, y_pred_b, average=None)[1],
            "matthews_corrcoef": matthews_corrcoef(y_true, y_pred_b),
            f"lift at {lift_at}": np.round(
                ClassificationEvaluator
                ._compute_lift(
                    y_true=y_true,
                    y_pred=y_pred,
                    lift_at=lift_at
                ), 2)
        })

    def plot_roc_curve(self, path: str = None, dim: tuple = (12, 8)):
        """Plot ROC curve of the model.

        Parameters
        ----------
        path : str, optional
            Path to store the figure.
        dim : tuple, optional
            Tuple with width and length of the plot.

        Raises
        ----------
        NotFittedError
            The instance is not fitted yet.
        """
        if self.roc_curve is None:
            msg = (
                "This {} instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this method."
            )
            raise NotFittedError(msg.format(self.__class__.__name__))

        auc = float(self.scalar_metrics.loc["AUC"])

        with plt.style.context("seaborn-whitegrid"):
            fig, ax = plt.subplots(figsize=dim)  # pylint: disable=unused-variable
            ax.plot(self.roc_curve["fpr"],
                    self.roc_curve["tpr"],
                    color="cornflowerblue", linewidth=3,
                    label=f"ROC curve (area = {auc:.3})")

            ax.plot([0, 1], [0, 1], color="darkorange", linewidth=3,
                    linestyle="--")
            ax.set_xlabel("False Positive Rate", fontsize=15)
            ax.set_ylabel("True Positive Rate", fontsize=15)
            ax.legend(loc="lower right")
            ax.set_title("ROC curve", fontsize=20)

            if path:
                plt.savefig(path, format="png", dpi=300, bbox_inches="tight")

        plt.show()

    def plot_confusion_matrix(
        self, path: str = None,
        dim: tuple = (12, 8),
        labels: list = None
    ):
        """Plot the confusion matrix.

        Parameters
        ----------
        path : str, optional
            Path to store the figure.
        dim : tuple, optional
            Tuple with width and length of the plot.
        labels : list, optional
            Optional list of labels, default "0" and "1".

        Raises
        ----------
        NotFittedError
            The instance is not fitted yet.
        """
        labels = labels or DEFAULT_LABELS
        if self.confusion_matrix is None:
            msg = (
                "This {} instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this method."
            )
            raise NotFittedError(msg.format(self.__class__.__name__))

        fig, ax = plt.subplots(figsize=dim)  # pylint: disable=unused-variable
        ax = sns.heatmap(
            self.confusion_matrix,
            annot=self.confusion_matrix.astype(str),
            fmt="s", cmap="Blues",
            xticklabels=labels, yticklabels=labels
        )
        ax.set_title("Confusion matrix", fontsize=20)

        if path:
            plt.savefig(path, format="png", dpi=300, bbox_inches="tight")

        plt.show()

    def plot_cumulative_response_curve(self, path: str = None, dim: tuple = (12, 8)):
        """Plot cumulative response curve.

        Parameters
        ----------
        path : str, optional
            Path to store the figure.
        dim : tuple, optional
            Tuple with width and length of the plot.

        Raises
        ----------
        NotFittedError
            The instance is not fitted yet.
        """
        if self.lift_curve is None:
            msg = (
                "This {} instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this method."
            )
            raise NotFittedError(msg.format(self.__class__.__name__))

        x_labels, lifts, inc_rate = self.lift_curve
        lifts = np.array(lifts)*inc_rate*100

        with plt.style.context("seaborn-ticks"):
            fig, ax = plt.subplots(figsize=dim)  # pylint: disable=unused-variable

            plt.bar(
                x_labels[::-1],
                lifts,
                align="center",
                color="cornflowerblue")
            plt.ylabel("response (%)", fontsize=16)
            plt.xlabel("decile", fontsize=16)
            ax.set_xticks(x_labels)
            ax.set_xticklabels(x_labels)

            plt.axhline(
                y=inc_rate*100,
                color="darkorange",
                linestyle="--",
                xmin=0.05,
                xmax=0.95,
                linewidth=3,
                label="Incidence"
            )

            # Legend
            ax.legend(loc="upper right")

            # Set Axis - make them pretty
            sns.despine(ax=ax, right=True, left=True)

            # Remove white lines from the second axis
            ax.grid(False)

            # Description
            ax.set_title("Cumulative Response curve", fontsize=20)

            if path is not None:
                plt.savefig(path, format="png", dpi=300, bbox_inches="tight")

            plt.show()

    def plot_lift_curve(self, path: str = None, dim: tuple = (12, 8)):
        """Plot lift per decile.

        Parameters
        ----------
        path : str, optional
            Path to store the figure.
        dim : tuple, optional
            Tuple with width and length of the plot.

        Raises
        ----------
        NotFittedError
            The instance is not fitted yet.
        """
        if self.lift_curve is None:
            msg = (
                "This {} instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this method."
            )
            raise NotFittedError(msg.format(self.__class__.__name__))

        x_labels, lifts, _ = self.lift_curve

        with plt.style.context("seaborn-ticks"):
            fig, ax = plt.subplots(figsize=dim)  # pylint: disable=unused-variable

            plt.bar(x_labels[::-1], lifts, align="center",
                    color="cornflowerblue")
            plt.ylabel("lift", fontsize=16)
            plt.xlabel("decile", fontsize=16)
            ax.set_xticks(x_labels)
            ax.set_xticklabels(x_labels)

            plt.axhline(
                y=1,
                color="darkorange",
                linestyle="--",
                xmin=0.05,
                xmax=0.95,
                linewidth=3,
                label="Baseline"
            )

            # Legend
            ax.legend(loc="upper right")

            # Set Axis - make them pretty
            sns.despine(ax=ax, right=True, left=True)

            # Remove white lines from the second axis
            ax.grid(False)

            # Description
            ax.set_title("Cumulative Lift curve", fontsize=20)

            if path is not None:
                plt.savefig(path, format="png", dpi=300, bbox_inches="tight")

            plt.show()

    def plot_cumulative_gains(self, path: str = None, dim: tuple = (12, 8)):
        """Plot cumulative gains per decile.

        Parameters
        ----------
        path : str, optional
            Path to store the figure.
        dim : tuple, optional
            Tuple with width and length of the plot.
        """
        with plt.style.context("seaborn-whitegrid"):
            fig, ax = plt.subplots(figsize=dim)  # pylint: disable=unused-variable

            ax.plot(self.cumulative_gains[0]*100, self.cumulative_gains[1]*100,
                    color="cornflowerblue", linewidth=3,
                    label="cumulative gains")
            ax.plot(ax.get_xlim(), ax.get_ylim(), linewidth=3,
                    ls="--", color="darkorange", label="random selection")

            ax.set_title("Cumulative Gains curve", fontsize=20)

            # Format axes
            ax.set_xlim([0, 100])
            ax.set_ylim([0, 105])

            # Format ticks
            ticks_loc_y = ax.get_yticks().tolist()
            ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc_y))
            ax.set_yticklabels([f"{x:3.0f}%" for x in ticks_loc_y])

            ticks_loc_x = ax.get_xticks().tolist()
            ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc_x))
            ax.set_xticklabels([f"{x:3.0f}%" for x in ticks_loc_x])

            # Legend
            ax.legend(loc="lower right")

            if path is not None:
                plt.savefig(path, format="png", dpi=300, bbox_inches="tight")
            plt.show()

    @staticmethod
    def _find_optimal_cutoff(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """Find the optimal probability cut off point for a classification model.

        Parameters
        ----------
        y_true : np.ndarray
            True binary target data labels.
        y_pred : np.ndarray
            Target scores of the model.

        Returns
        -------
        float
            Optimal cut-off probability for the model.
        """
        fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_pred)
        return ClassificationEvaluator._compute_optimal_cutoff(fpr, tpr, thresholds)

    @staticmethod
    def _compute_optimal_cutoff(
        fpr: np.ndarray,
        tpr: np.ndarray,
        thresholds: np.ndarray
    ) -> float:
        """Calculate the optimal probability cut-off point for a classification model.

        The optimal cut-off would be where TPR is high and FPR is low, hence
        TPR - (1-FPR) should be zero or close to zero for the optimal cut-off.

        Parameters
        ----------
        fpr : np.ndarray
            False positive rate for various thresholds.
        tpr : np.ndarray
            True positive rate for various thresholds.
        thresholds : np.ndarray
            List of thresholds for which fpr and tpr were computed.

        Returns
        -------
        float
            Optimal probability cut-off point.
        """
        temp = np.absolute(tpr - (1-fpr))

        # index for optimal value is the one for which temp is minimal
        optimal_index = np.where(temp == min(temp))[0]

        return thresholds[optimal_index][0]

    @staticmethod
    def _compute_cumulative_gains(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> tuple:
        """Compute cumulative gains of the model.

        Code from (https://github.com/reiinakano/scikit-plot/blob/
                   2dd3e6a76df77edcbd724c4db25575f70abb57cb/
                   scikitplot/helpers.py#L157)

        Parameters
        ----------
        y_true : np.ndarray
            True binary target data labels.
        y_pred : np.ndarray
            Target scores of the model.

        Returns
        -------
        tuple
            With x-labels, and gains.
        """
        # make y_true a boolean vector
        y_true = (y_true == 1)

        sorted_indices = np.argsort(y_pred)[::-1]
        y_true = y_true[sorted_indices]
        gains = np.cumsum(y_true)

        percentages = np.arange(start=1, stop=len(y_true) + 1)

        gains = gains / float(np.sum(y_true))
        percentages = percentages / float(len(y_true))

        gains = np.insert(gains, 0, [0])
        percentages = np.insert(percentages, 0, [0])

        return percentages, gains

    @staticmethod
    def _compute_lift_per_bin(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bins: int = 10
    ) -> tuple:
        """Compute lift of the model for a given number of bins.

        Parameters
        ----------
        y_true : np.ndarray
            True binary target data labels.
        y_pred : np.ndarray
            Target scores of the model.
        n_bins : int, optional
            Defines the number of bins used to calculate the lift curve for
            (by default 10, so deciles).

        Returns
        -------
        tuple
            Includes x-labels, lifts per decile, and target incidence.
        """
        lifts = [
            ClassificationEvaluator
            ._compute_lift(
                y_true=y_true,
                y_pred=y_pred,
                lift_at=perc_lift
            )
            for perc_lift in np.linspace(1/n_bins, 1, num=n_bins, endpoint=True)
        ]

        x_labels = [len(lifts)-x for x in np.arange(0, len(lifts), 1)]

        return x_labels, lifts, y_true.mean()

    @staticmethod
    def _compute_lift(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        lift_at: float = 0.05
    ) -> float:
        """Calculate lift on a specified level.

        Parameters
        ----------
        y_true : np.ndarray
            True binary target data labels.
        y_pred : np.ndarray
            Target scores of the model.
        lift_at : float, optional
            At what top level percentage the lift should be computed.

        Returns
        -------
        float
            Lift of the model.
        """
        # Make sure it is numpy array
        y_true_ = np.array(y_true)
        y_pred_ = np.array(y_pred)

        # Make sure it has correct shape
        y_true_ = y_true_.reshape(len(y_true_), 1)
        y_pred_ = y_pred_.reshape(len(y_pred_), 1)

        # Merge data together
        y_data = np.hstack([y_true_, y_pred_])

        # Calculate necessary variables
        nrows = len(y_data)
        stop = int(np.floor(nrows*lift_at))
        avg_incidence = np.einsum("ij->j", y_true_)/float(len(y_true_))

        # Sort and filter data
        data_sorted = (
            y_data[y_data[:, 1].argsort()[::-1]][:stop, 0]
            .reshape(stop, 1)
        )

        # Calculate lift (einsum is a very fast way of summing, but needs specific shape)
        inc_in_top_n = np.einsum("ij->j", data_sorted)/float(len(data_sorted))
        lift = np.round(inc_in_top_n/avg_incidence, 2)[0]
        return lift


class RegressionEvaluator():
    """Evaluator class encapsulating regression model metrics and plotting functionality.

    Attributes
    ----------
    y_true : np.ndarray
        True binary target data labels.
    y_pred : np.ndarray
        Target scores of the model.
    scalar_metrics : dict
        Map containing various scalar evaluation metrics (R-squared, MAE, MSE, RMSE)
    qq : pd.Series
        Theoretical quantiles and associated actual residuals.
    """

    def __init__(self):
        """Initialize the RegressionEvaluator."""
        self.y_true = None
        self.y_pred = None

        # Placeholder to store fitted output
        self.scalar_metrics = None
        self.qq = None

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Fit the evaluator by computing the relevant evaluation metrics on the inputs.

        Parameters
        ----------
        y_true : np.ndarray
            True labels.
        y_pred : np.ndarray
            Model scores.
        """
        # Compute the various evaluation metrics
        self.scalar_metrics = RegressionEvaluator._compute_scalar_metrics(y_true, y_pred)

        self.y_true = y_true
        self.y_pred = y_pred

        # Compute qq info
        self.qq = RegressionEvaluator._compute_qq_residuals(y_true, y_pred)

    @staticmethod
    def _compute_scalar_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> pd.Series:
        """Compute various scalar performance measures.

        Parameters
        ----------
        y_true : np.ndarray
            True binary target data labels.
        y_pred : np.ndarray
            Target scores of the model.

        Returns
        -------
        pd.Series
            Contains various performance measures of the model, being:
                R-squared (coefficient of determination, usually denoted as R-squared)
                Mean absolute error (expected value of the absolute error loss)
                Mean squared error (expected value of the quadratic error)
                Root mean squared error (sqrt of expected value of the quadratic error)
        """
        return pd.Series({
            "R2": r2_score(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mean_squared_error(y_true, y_pred),
            "RMSE": sqrt(mean_squared_error(y_true, y_pred))
        })

    @staticmethod
    def _compute_qq_residuals(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> pd.Series:
        """Compute various scalar performance measures.

        Parameters
        ----------
        y_true : np.ndarray
            True binary target data labels.
        y_pred : np.ndarray
            Target scores of the model.

        Returns
        -------
        pd.Series
            Theoretical quantiles and associated actual residuals.
        """
        # also possible directly via statsmodels.api.qqplot()

        n = len(y_true)

        df = pd.DataFrame({"res": sorted((y_true - y_pred))})  # ascending order
        m, s = df["res"].mean(), df["res"].std()

        df["z_res"] = df["res"].apply(lambda x: (x-m)/s)
        df["rank"] = df.index+1
        df["percentile"] = df["rank"].apply(lambda x: x/(n+1))  # divide by n+1 to avoid inf
        df["q_theoretical"] = norm.ppf(df["percentile"])

        return pd.Series({
            "quantiles": df["q_theoretical"].values,
            "residuals": df["z_res"].values,
        })

    def plot_predictions(self, path: str = None, dim: tuple = (12, 8)):
        """Plot predictions from the model against actual values.

        Parameters
        ----------
        path : str, optional
            Path to store the figure.
        dim : tuple, optional
            Tuple with width and length of the plot.

        Raises
        ----------
        NotFittedError
            The instance is not fitted yet.
        """
        if self.y_true is None and self.y_pred is None:
            msg = (
                "This {} instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this method."
            )
            raise NotFittedError(msg.format(self.__class__.__name__))

        y_true = self.y_true
        y_pred = self.y_pred

        with plt.style.context("seaborn-whitegrid"):
            fig, ax = plt.subplots(figsize=dim)  # pylint: disable=unused-variable

            x = np.arange(1, len(y_true)+1)

            ax.plot(x, y_true, ls="--", label="actuals", color="darkorange", linewidth=3)
            ax.plot(x, y_pred, label="predictions", color="cornflowerblue", linewidth=3)

            ax.set_xlabel("Index", fontsize=15)
            ax.set_ylabel("Value", fontsize=15)
            ax.legend(loc="best")
            ax.set_title("Predictions vs. Actuals", fontsize=20)

            if path:
                plt.savefig(path, format="png", dpi=300, bbox_inches="tight")

        plt.show()

    def plot_qq(self, path: str = None, dim: tuple = (12, 8)):
        """Display a Q-Q plot from the standardized prediction residuals.

        Parameters
        ----------
        path : str, optional
            Path to store the figure.
        dim : tuple, optional
            Tuple with width and length of the plot.

        Raises
        ----------
        NotFittedError
            The instance is not fitted yet.
        """
        if self.qq is None:
            msg = (
                "This {} instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this method."
            )
            raise NotFittedError(msg.format(self.__class__.__name__))

        with plt.style.context("seaborn-whitegrid"):
            fig, ax = plt.subplots(figsize=dim)  # pylint: disable=unused-variable

            x = self.qq["quantiles"]
            y = self.qq["residuals"]

            ax.plot(x, x, ls="--", label="perfect model", color="darkorange", linewidth=3)
            ax.plot(x, y, label="current model", color="cornflowerblue", linewidth=3)

            ax.set_xlabel("Theoretical quantiles", fontsize=15)
            ax.set_xticks(range(int(np.floor(min(x))), int(np.ceil(max(x[x < float("inf")])))+1, 1))

            ax.set_ylabel("Standardized residuals", fontsize=15)
            ax.set_yticks(range(int(np.floor(min(y))), int(np.ceil(max(y[x < float("inf")])))+1, 1))

            ax.legend(loc="best")
            ax.set_title("Q-Q plot", fontsize=20)

            if path:
                plt.savefig(path, format="png", dpi=300, bbox_inches="tight")

        plt.show()
