import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import NotFittedError


class Evaluator():

    """Summary

    Attributes
    ----------
    confusion_matrix : np.ndarray
        Confusion matrix computed for a particular cut-off
    cumulative_gains : tuple
        data for plotting cumulative gains curve
    evaluation_metrics : dict
        map containing various scalar evaluation metics such as AUC, ...
    lift_at : float
        parameter to determine at which top level percentage the lift of the
        model should be computed
    lift_curve : tuple
        data for plotting lift curve(s)
    probability_cutoff : float
        probability cut off to convert probability scores to a binary score
    roc_curve : dict
        map containing true-positive-rate, false-positve-rate at various
        thresholds (also incl.)
    """

    def __init__(self, probability_cutoff: float=None,
                 lift_at: float=0.05):

        self.lift_at = lift_at
        self.probability_cutoff = probability_cutoff

        # Placeholder to store fitted output
        self.evaluation_metrics = None
        self.roc_curve = None
        self.confusion_matrix = None
        self.lift_curve = None
        self.cumulative_gains = None

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray):

        fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_pred)

        # if probability_cutoff is not set, take the optimal cut off
        if not self.probability_cutoff:
            self.probability_cutoff = (Evaluator.
                                       _compute_optimal_cutoff(fpr, tpr,
                                                               thresholds))

        # Transform probabilities to binary array using cut off:
        y_pred_b = np.array([0 if pred <= self.probability_cutoff else 1
                             for pred in y_pred])

        # Compute the various evaluation metrics
        self.evaluation_metrics = Evaluator.compute_evaluation_metrics(
            y_true,
            y_pred,
            y_pred_b,
            self.lift_at
        )

        self.roc_curve = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}
        self.confusion_matrix = confusion_matrix(y_true, y_pred_b)
        self.lift_curve = Evaluator._compute_lift_per_decile(y_true, y_pred)
        self.cumulative_gains = Evaluator._compute_cumulative_gains(y_true,
                                                                    y_pred)

    def get_metrics(self) -> pd.Series:
        """Get the evaluation_metrics attribute as a pandas Series

        Returns
        -------
        pd.Series
            Score of various scalar evaluation metrics for the model
        """
        return pd.Series(self.evaluation_metrics)

    @staticmethod
    def compute_evaluation_metrics(y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   y_pred_b: np.ndarray,
                                   lift_at: float) -> dict:
        """Convenient function to compute various performance measures and
        return them in a dict

        Parameters
        ----------
        y_true : np.ndarray
            True binary target data labels
        y_pred : np.ndarray
            Target scores of the model
        y_pred_b : np.ndarray
            Predicted target data labels (binary)
        lift_at : float
            At what top level percentage the lift should be computed

        Returns
        -------
        dict
            contains various performance measures of the model
        """
        return {
            "accuracy": accuracy_score(y_true, y_pred_b),
            "AUC": roc_auc_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred_b),
            "recall": recall_score(y_true, y_pred_b),
            "F1": f1_score(y_true, y_pred_b, average=None)[1],
            f"lift at {lift_at}": np.round(Evaluator
                                           ._compute_lift(y_true=y_true,
                                                          y_pred=y_pred,
                                                          lift_at=lift_at), 2)
        }

    def plot_roc_curve(self, path: str=None, dim: tuple=(12, 8)):
        """Plot ROC curves of the model

        Parameters
        ----------
        path : str, optional
            path to store the figure
        dim : tuple, optional
            tuple with width and lentgh of the plot
        """

        if self.roc_curve is None:
            msg = ("This {} instance is not fitted yet. Call 'fit' with "
                   "appropriate arguments before using this method.")

            raise NotFittedError(msg.format(self.__class__.__name__))

        auc = self.evaluation_metrics["AUC"]

        fig, ax = plt.subplots(figsize=dim)

        ax.plot(self.roc_curve["fpr"],
                self.roc_curve["tpr"],
                color="darkorange", lw=2,
                label="ROC curve (area = {s:.3})".format(s=auc))

        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        ax.set_xlabel("False Positive Rate", fontsize=15)
        ax.set_ylabel("True Positive Rate", fontsize=15)
        ax.legend(loc="lower right")
        ax.set_title("ROC Curve", fontsize=20)

        if path:
            plt.savefig(path, format="png", dpi=300, bbox_inches="tight")

        plt.show()

    def plot_confusion_matrix(self, path: str=None, dim: tuple=(12, 8),
                              labels: list=["0", "1"]):
        """Plot the confusion matrix

        Parameters
        ----------
        path : str, optional
            path to store the figure
        dim : tuple, optional
            tuple with width and lentgh of the plot
        labels : list, optional
            Optional list of labels, default "0" and "1"
        """

        if self.confusion_matrix is None:
            msg = ("This {} instance is not fitted yet. Call 'fit' with "
                   "appropriate arguments before using this method.")

            raise NotFittedError(msg.format(self.__class__.__name__))

        fig, ax = plt.subplots(figsize=dim)
        ax = sns.heatmap(self.confusion_matrix,
                         annot=self.confusion_matrix.astype(str),
                         fmt="s", cmap="Reds",
                         xticklabels=labels, yticklabels=labels)
        ax.set_title("Confusion matrix", fontsize=20)

        if path:
            plt.savefig(path, format="png", dpi=300, bbox_inches="tight")

        plt.show()

    def plot_cumulative_response_curve(self, path: str=None,
                                       dim: tuple=(12, 8)):
        """Plot cumulative response curve

        Parameters
        ----------
        path : str, optional
            path to store the figure
        dim : tuple, optional
            tuple with width and lentgh of the plot
        """

        if self.lift_curve is None:
            msg = ("This {} instance is not fitted yet. Call 'fit' with "
                   "appropriate arguments before using this method.")

            raise NotFittedError(msg.format(self.__class__.__name__))

        x_labels, lifts, inc_rate = self.lift_curve

        lifts = np.array(lifts)*inc_rate*100

        fig, ax = plt.subplots(figsize=dim)
        plt.style.use("default")

        plt.bar(x_labels[::-1], lifts, align="center", color="cornflowerblue")
        plt.ylabel("response (%)", fontsize=16)
        plt.xlabel("decile", fontsize=16)
        ax.set_xticks(x_labels)
        ax.set_xticklabels(x_labels)

        plt.axhline(y=inc_rate*100, color="darkorange", linestyle="--",
                    xmin=0.05, xmax=0.95, linewidth=3, label="Incidence")

        #Legend
        ax.legend(loc="upper right")

        ##Set Axis - make them pretty
        sns.despine(ax=ax, right=True, left=True)

        #Remove white lines from the second axis
        ax.grid(False)

        ##Description
        ax.set_title("Cumulative response", fontsize=20)

        if path is not None:
            plt.savefig(path, format="png", dpi=300, bbox_inches="tight")

        plt.show()

    def plot_lift_curve(self, path: str=None, dim: tuple=(12, 8)):
        """Plot lift per decile

        Parameters
        ----------
        path : str, optional
            path to store the figure
        dim : tuple, optional
            tuple with width and lentgh of the plot
        """

        if self.lift_curve is None:
            msg = ("This {} instance is not fitted yet. Call 'fit' with "
                   "appropriate arguments before using this method.")

            raise NotFittedError(msg.format(self.__class__.__name__))

        x_labels, lifts, _ = self.lift_curve

        fig, ax = plt.subplots(figsize=dim)
        plt.style.use("default")

        plt.bar(x_labels[::-1], lifts, align="center", color="cornflowerblue")
        plt.ylabel("lift", fontsize=16)
        plt.xlabel("decile", fontsize=16)
        ax.set_xticks(x_labels)
        ax.set_xticklabels(x_labels)

        plt.axhline(y=1, color="darkorange", linestyle="--",
                    xmin=0.05, xmax=0.95, linewidth=3, label="Baseline")

        #Legend
        ax.legend(loc="upper right")

        ##Set Axis - make them pretty
        sns.despine(ax=ax, right=True, left=True)

        #Remove white lines from the second axis
        ax.grid(False)

        ##Description
        ax.set_title("Cumulative Lift", fontsize=20)

        if path is not None:
            plt.savefig(path, format="png", dpi=300, bbox_inches="tight")

        plt.show()

    def plot_cumulative_gains(self, path: str=None, dim: tuple=(12, 8)):
        """Plot lift per decile

        Parameters
        ----------
        path : str, optional
            path to store the figure
        dim : tuple, optional
            tuple with width and lentgh of the plot
        """
        pass

    @staticmethod
    def find_optimal_cutoff(y_true: np.ndarray,
                            y_pred: np.ndarray) -> float:
        """Find the optimal probability cut off point for a
        classification model. Wrapper around _compute_optimal_cutoff

        Parameters
        ----------
        y_true : np.ndarray
            True binary target data labels
        y_pred : np.ndarray
            Target scores of the model

        Returns
        -------
        float
            Optimal cut off probability for the model
        """
        return Evaluator._compute_optimal_cutoff(roc_curve(y_true=y_true,
                                                           y_score=y_pred))

    @staticmethod
    def _compute_optimal_cutoff(fpr: np.ndarray, tpr: np.ndarray,
                                thresholds: np.ndarray) -> float:
        """Find the optimal probability cut off point for a
        classification model

        Parameters
        ----------
        fpr : np.ndarray
            false positive rate for various thresholds
        tpr : np.ndarray
            true positive rate for various thresholds
        thresholds : np.ndarray
            list of thresholds for which fpr and tpr were computed

        Returns
        -------
        float
            Description
        """

        # The optimal cut off would be where tpr is high and fpr is low, hence
        # tpr - (1-fpr) should be zero or close to zero for the optimal cut off
        temp = np.absolute(tpr - (1-fpr))

        # index for optimal value is the one for which temp is minimal
        optimal_index = np.where(temp == min(temp))[0]

        return thresholds[optimal_index][0]

    @staticmethod
    def _compute_cumulative_gains(y_true: np.ndarray,
                                  y_pred: np.ndarray) -> tuple:
        """Compute lift of the model per decile, returns x-labels, lifts and
        the target incidence to create cummulative response curves

        Parameters
        ----------
        y_true : np.ndarray
            True binary target data labels
        y_pred : np.ndarray
            Target scores of the model

        Returns
        -------
        tuple
            x-labels, lifts per decile and target incidence
        """
        pass

    @staticmethod
    def _compute_lift_per_decile(y_true: np.ndarray,
                                 y_pred: np.ndarray) -> tuple:
        """Compute lift of the model per decile, returns x-labels, lifts and
        the target incidence to create cummulative response curves

        Parameters
        ----------
        y_true : np.ndarray
            True binary target data labels
        y_pred : np.ndarray
            Target scores of the model

        Returns
        -------
        tuple
            x-labels, lifts per decile and target incidence
        """

        lifts = [Evaluator._compute_lift(y_true=y_true,
                                         y_pred=y_pred,
                                         lift_at=perc_lift)
                 for perc_lift in np.arange(0.1, 1.1, 0.1)]

        x_labels = [len(lifts)-x for x in np.arange(0, len(lifts), 1)]

        return x_labels, lifts, y_true.mean()

    @staticmethod
    def _compute_lift(y_true: np.ndarray, y_pred: np.ndarray,
                      lift_at: float=0.05) -> float:
        """Calculates lift given two arrays on specified level
           %timeit
           50.3 µs ± 1.94 µs per loop (mean ± std. dev. of 7 runs,
                                        10000 loops each)

        Parameters
        ----------
        y_true : np.ndarray
            True binary target data labels
        y_pred : np.ndarray
            Target scores of the model
        lift_at : float, optional
            At what top level percentage the lift should be computed

        Returns
        -------
        float
            lift of the model
        """

        #Make sure it is numpy array
        y_true_ = np.array(y_true)
        y_pred_ = np.array(y_pred)

        #Make sure it has correct shape
        y_true_ = y_true_.reshape(len(y_true_), 1)
        y_pred_ = y_pred_.reshape(len(y_pred_), 1)

        #Merge data together
        y_data = np.hstack([y_true_, y_pred_])

        #Calculate necessary variables
        nrows = len(y_data)
        stop = int(np.floor(nrows*lift_at))
        avg_incidence = np.einsum("ij->j", y_true_)/float(len(y_true_))

        #Sort and filter data
        data_sorted = (y_data[y_data[:, 1].argsort()[::-1]][:stop, 0]
                       .reshape(stop, 1))

        #Calculate lift (einsum is very fast way of summing,
        # needs specific shape)
        inc_in_top_n = np.einsum("ij->j", data_sorted)/float(len(data_sorted))

        lift = np.round(inc_in_top_n/avg_incidence, 2)[0]

        return lift
