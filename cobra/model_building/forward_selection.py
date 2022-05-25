
import logging
from typing import Callable, Optional

import pandas as pd
from tqdm.auto import tqdm

from cobra.model_building import LogisticRegressionModel, LinearRegressionModel

log = logging.getLogger(__name__)

class ForwardFeatureSelection:
    """Perform forward feature selection for a given dataset using a given
    algorithm.

    Predictors are sequentially added to the model, starting with the one that
    has the highest univariate predictive power, and then proceeding with those
    that jointly lead to the best fit, optimizing (tuning) for model
    performance on the selection set, measured with AUC (default for
    classification), RMSE (default for regression) or a custom metric (when
    passing the metric parameter and possibly also metric_args and
    metric_kwargs.)

    Interaction effects are not explicitly modeled, yet they are implicitly
    present given the feature selection and the underlying feature
    correlation structure.

    Attributes
    ----------
    model_type : str
        Model type (``classification`` or ``regression``).
    MLModel: Cobra model
        LogisticRegressionModel or LinearRegressionModel.
    max_predictors : int
        Maximum number of predictors allowed in any model. This corresponds
        more or less with the maximum number of steps in the forward feature
        selection.
    pos_only : bool
        Whether or not the model coefficients should all be positive (no sign flips).
    self._fitted_models : list
        List of fitted models.
    metric : Callable (function), optional
        Function that evaluates the model's performance, by calculating a
        certain evaluation metric.
        For more details about the possibilities here, refer to the
        documentation of the metric parameter in the evaluate() function of
        either models.LogisticRegressionModel or models.LinearRegressionModel,
        depending on which model you are going to use in this forward feature
        selection.
    metric_args : dict, optional
        Arguments (for example: lift_at=0.05) to be passed to the metric
        function when evaluating the model's performance.
        Example metric function in which this is required:
        ClassificationEvaluator._compute_lift(y_true=y_true,
                                              y_score=y_score,
                                              lift_at=0.05)
    metric_kwargs : dict, optional
        Keyword arguments (for example: normalize=True) to be passed to the
        metric function when evaluating the model's performance.
        Example metric function in which this is required (from
        scikit-learn):
        def accuracy_score(y_true, y_pred, *, normalize=True, sample_weight=None)
    higher_is_better : bool, optional
        Whether the model is performing better if the chosen evaluation
        metric results in a higher score (higher_is_better=True),
        or worse (higher_is_better=False, meaning "lower is better").
    """

    def __init__(self,
                 model_type: str="classification",
                 max_predictors: int=50,
                 pos_only: bool=True,
                 metric: Optional[Callable] = None,
                 metric_args: Optional[dict] = None,
                 metric_kwargs: Optional[dict] = None,
                 higher_is_better: Optional[bool] = None):

        self.model_type = model_type
        if model_type == "classification":
            self.MLModel = LogisticRegressionModel
        elif model_type == "regression":
            self.MLModel = LinearRegressionModel

        self.max_predictors = max_predictors
        self.pos_only = pos_only

        if higher_is_better is None:
            if metric is None:
                if self.MLModel == LogisticRegressionModel:
                    # If no custom evaluation metric is chosen,
                    # the LogisticRegressionModel uses AUC as default metric,
                    # so "higher is better" evaluation logic is applied on the
                    # evaluation scores.
                    self.higher_is_better = True
                elif self.MLModel == LinearRegressionModel:
                    # If no custom evaluation metric is chosen,
                    # the LinearRegressionModel uses RMSE as default metric,
                    # so "lower is better" evaluation logic is applied on the
                    # evaluation scores.
                    self.higher_is_better = False
                else:
                    raise ValueError("The configured machine learning model is "
                                     "not the standard logistic regression or "
                                     "linear regression model. "
                                     "Therefore, please fill the metric and "
                                     "higher_is_better arguments.")
            else:
                raise ValueError("You chose a custom evaluation metric. "
                                 "Please fill the higher_is_better argument.")
        else:
            self.higher_is_better = higher_is_better

        self.metric = metric
        self.metric_args = metric_args
        self.metric_kwargs = metric_kwargs


        self._fitted_models = []

    def get_model_from_step(self, step: int):
        """Get fitted model from a particular step.

        Parameters
        ----------
        step : int
            Particular step in the forward selection.

        Returns
        -------
        self.MLModel
            Fitted model from the given step.

        Raises
        ------
        ValueError
            In case step is larger than the number of available models.
        """
        if len(self._fitted_models) <= step:
            raise ValueError(f"No model available for step {step}. "
                             "The first step starts from index 0.")

        return self._fitted_models[step]

    def compute_model_performances(self, data: pd.DataFrame,
                                   target_column_name: str,
                                   splits: list=["train", "selection", "validation"]
                                   ) -> pd.DataFrame:
        """Compute for each model the performance for different sets (e.g.
        train-selection-validation) and return them along with a list of
        predictors used in the model. Note that the computation of the
        performance for each split is cached inside the model itself, so it
        is inexpensive to perform it multiple times!

        Parameters
        ----------
        data : pd.DataFrame
            Dataset for which to compute performance of each model.
        target_column_name : str
            Name of the target column.
        splits : list, optional
            List of splits to compute performance on.

        Returns
        -------
        DatFrame
            Contains for each model the performance for train, selection and
            validation sets as well as the set of predictors used in this model.
        """
        results = []
        predictor_set = set([])

        for model in self._fitted_models:
            last_added_predictor = (set(model.predictors)
                                    .difference(predictor_set))
            tmp = {
                "predictors": model.predictors,
                "last_added_predictor": list(last_added_predictor)[0]
            }

            # Evaluate model on each dataset split,
            # e.g. train-selection-validation
            tmp.update({
                f"{split}_performance": model.evaluate(
                    data[data["split"] == split],
                    data[data["split"] == split][target_column_name],
                    split=split,  # parameter used for caching
                    metric=self.metric,
                    metric_args=self.metric_args,
                    metric_kwargs=self.metric_kwargs)
                for split in splits
            })

            results.append(tmp)

            predictor_set = predictor_set.union(set(model.predictors))

        df = pd.DataFrame(results)
        df["model_type"] = self.model_type

        return df

    def fit(self, train_data: pd.DataFrame, target_column_name: str,
            predictors: list, forced_predictors: list=[],
            excluded_predictors: list=[]):
        """Fit the forward feature selection estimator.

        Parameters
        ----------
        train_data : pd.DataFrame
            Data on which to fit the model. Should include a "train"
            and "selection" split for correct model selection! The
            "train" split is used to train a model, the "selection"
            split is used to evaluate which model to include in the
            actual forward feature selection.
        target_column_name : str
            Name of the target column.
        predictors : list
            List of predictors on which to train the estimator.
        forced_predictors : list, optional
            List of predictors to force in the estimator.
        excluded_predictors : list, optional
            List of predictors to exclude from the estimator.

        Raises
        ------
        ValueError
            In case the number of forced predictors is larger than the maximum
            number of allowed predictors in the model.
        """

        assert "split" in train_data.columns, "The train_data input df does not include a split column."
        assert len(set(["train", "selection"]).difference(set(train_data["split"].unique()))) == 0, \
            "The train_data input df does not include a 'train' and 'selection' split."

        # remove excluded predictors from predictor lists
        filtered_predictors = [var for var in predictors
                               if (var not in excluded_predictors and
                                   var not in forced_predictors)]

        # checks on predictor lists and self.max_predictors attr
        if len(forced_predictors) > self.max_predictors:
            raise ValueError("Size of forced_predictors cannot be bigger than "
                             "max_predictors.")
        elif len(forced_predictors) == self.max_predictors:
            log.info("Size of forced_predictors equals max_predictors "
                     "only one model will be trained...")
            # train model with all forced_predictors (only)
            (self._fitted_models
             .append(self._train_model(train_data[train_data["split"] == "train"],
                                       target_column_name,
                                       forced_predictors)))
        else:
            self._fitted_models = self._forward_selection(train_data,
                                                          target_column_name,
                                                          filtered_predictors,
                                                          forced_predictors)

    def _forward_selection(self,
                           train_data: pd.DataFrame,
                           target_column_name: str,
                           predictors: list,
                           forced_predictors: list = []) -> list:
        """Perform the forward feature selection algorithm to compute a list
        of models (with increasing performance). The length of the list,
        i.e. the number of models, is bounded by the max_predictors class
        attribute.

        Parameters
        ----------
        train_data : pd.DataFrame
            Data on which to fit the model.
        target_column_name : str
            Name of the target column.
        predictors : list
            List of predictors on which to train the models.
        forced_predictors : list, optional
            List of predictors to force in the models.

        Returns
        -------
        list
            List of fitted models where the index of the list indicates the
            number of predictors minus one (as indices start from 0).
        """
        fitted_models = []
        current_predictors = []

        max_steps = 1 + min(self.max_predictors,
                            len(predictors) + len(forced_predictors))

        for step in tqdm(range(1, max_steps), desc="Sequentially adding best "
                                                   "predictor..."):
            if step <= len(forced_predictors):
                # first, we go through the forced predictors
                candidate_predictors = [var for var in forced_predictors
                                        if var not in current_predictors]
            else:
                candidate_predictors = [var for var in (predictors
                                                        + forced_predictors)
                                        if var not in current_predictors]

            model = self._find_next_best_model(train_data,
                                               target_column_name,
                                               candidate_predictors,
                                               current_predictors)

            if model is not None:
                # Add new model predictors to the list of current predictors
                current_predictors = list(set(current_predictors)
                                          .union(set(model.predictors)))

                fitted_models.append(model)
            # else:
            #     # If model returns None for the first time,
            #     # one can in theory stop the feature selection process
            #     # but we leave it run such that tqdm cleanly finishes
            #     break

        if not fitted_models:
            log.error("No models found in forward selection.")

        return fitted_models

    def _find_next_best_model(self,
                              train_data: pd.DataFrame,
                              target_column_name: str,
                              candidate_predictors: list,
                              current_predictors: list):
        """Given a list of current predictors which are already selected to
        be include in the model, find amongst a list candidate predictors
        the predictor to add to the selected list so that the resulting model
        has the best performance.

        Parameters
        ----------
        train_data : pd.DataFrame
            Data on which to fit the model.
        target_column_name : str
            Name of the target column.
        candidate_predictors : list
            List of candidate predictors to test.
        current_predictors : list
            List of predictors on which to train the models.

        Returns
        -------
        self.MLModel
            Best performing model.
        """
        # placeholders
        best_model = None

        # Set the performance intially with the worst possible value,
        # depending on whether higher_is_better is true or false for the
        # chosen evaluation metric.
        if self.higher_is_better:
            best_performance = -float("inf")
        else:
            best_performance = float("inf")

        fit_data = train_data[train_data["split"] == "train"]  # data to fit the models with
        sel_data = train_data[train_data["split"] == "selection"]  # data to compare the models with

        for pred in candidate_predictors:
            # Train a model with an additional predictor
            model = self._train_model(fit_data, target_column_name,
                                      (current_predictors + [pred]))

            # Evaluate the model
            performance = (model
                           .evaluate(sel_data[current_predictors + [pred]],
                                     sel_data[target_column_name],
                                     split="selection",
                                     metric=self.metric,
                                     metric_args=self.metric_args,
                                     metric_kwargs=self.metric_kwargs))

            if self.pos_only and (not (model.get_coef() >= 0).all()):
                continue

            # Check if the model is better than the current best model
            # and if it is, replace the current best.
            if self.higher_is_better and performance > best_performance:
                best_performance = performance
                best_model = model
            elif not self.higher_is_better and performance < best_performance:
                best_performance = performance
                best_model = model

        return best_model

    def _train_model(self, train_data: pd.DataFrame, target_column_name: str,
                     predictors: list):
        """Train the model with a given set of predictors.

        Parameters
        ----------
        train_data : pd.DataFrame
            Data on which to fit the model.
        target_column_name : str
            Name of the target column.
        predictors : list
            List of predictors on which to train the models.

        Returns
        -------
        self.MLModel
            Trained model.
        """
        model = self.MLModel()

        model.fit(train_data[predictors], train_data[target_column_name])

        return model
