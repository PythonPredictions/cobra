import logging
log = logging.getLogger(__name__)

import pandas as pd

from cobra.model_building import LogisticRegressionModel as MLModel


class ForwardFeatureSelection:

    """Perform forward feature selection for a given dataset using a given
    model.

    Attributes
    ----------
    max_predictors : int
        maximum number of predictors allowed in any model. This corresponds
        more or less with the maximum number of steps in the forward feature
        selection
    model_name : str
        name of the model to use for forward feature selection
    pos_only : bool
        whether or not the model coefficients should all be positive
    """

    def __init__(self, max_predictors: int=50,
                 model_name: str="logistic-regression", pos_only: bool=True):

        self.pos_only = pos_only
        self.max_predictors = max_predictors
        self.model_name = model_name

        self._fitted_models = []

    def get_model_from_step(self, step: int) -> MLModel:
        """Get fitted model from a particular step

        Parameters
        ----------
        step : int
            Particular step in the forward selection

        Returns
        -------
        MLModel
            Fitted model from the given step

        Raises
        ------
        ValueError
            in case step is larger than the number of available models
        """
        if len(self._fitted_models) < step:
            raise ValueError(f"No model available for step {step}")

        return self._fitted_models[step]

    def compute_model_performances(self, data: pd.DataFrame,
                                   target_column_name: str) -> list:
        """Compute for each model the performance for
           train-selection-validation sets and return them along with a list
           of predictors used in the model.
           Note that the computation of the performance for each split is
           cached inside the model itself, so it is inexpensive to perform
           it multiple times!

        Parameters
        ----------
        data : pd.DataFrame
            dataset for which to compute performance of each model
        target_column_name : str
            name of the target column

        Returns
        -------
        list
            A list containing for each model the performance for
            train-selection-validation sets as well as the set of predictors
            used in this model
        """
        results = []
        predictor_set = set([])
        for model in self._fitted_models:
            # Evaluate model
            performance_train = model.evaluate(
                data[data["split"] == "train"],
                data[data["split"] == "train"][target_column_name],
                split="train"
                )

            performance_selection = model.evaluate(
                data[data["split"] == "selection"],
                data[data["split"] == "selection"][target_column_name],
                split="selection"
                )

            performance_validation = model.evaluate(
                data[data["split"] == "validation"],
                data[data["split"] == "validation"][target_column_name],
                split="validation"
                )

            last_added_predictor = (set(model.predictors)
                                    .difference(predictor_set))

            results.append({
                "predictors": model.predictors,
                "last_added_predictor": list(last_added_predictor)[0],
                "train_performance": performance_train,
                "selection_performance": performance_selection,
                "validation_performance": performance_validation
            })

            predictor_set = predictor_set.union(set(model.predictors))

        return results

    def fit(self, train_data: pd.DataFrame, target_column_name: str,
            predictors: list, forced_predictors: list=[],
            excluded_predictors: list=[]):
        """Summary

        Parameters
        ----------
        data : pd.DataFrame
            Description
        target_column_name : str
            Description
        predictors : list
            Description
        forced_predictors : list, optional
            Description
        excluded_predictors : list, optional
            Description

        Raises
        ------
        ValueError
            In case the number of forced predictors is larger than the maximum
            number of allowed predictors in the model
        """
        # remove excluded predictors from predictor lists
        filterd_predictors = [var for var in predictors
                              if var not in excluded_predictors]

        # checks on predictor lists and self.max_predictors attr
        if len(forced_predictors) > self.max_predictors:
            raise ValueError("Size or forced_predictors cannot be bigger than "
                             "max_predictors")
        elif len(forced_predictors) == self.max_predictors:
            log.info("Size of forced_predictors equals max_predictors "
                     "only one model will be trained...")
            # train model with all forced_predictors (only)
            (self._fitted_models
             .append(self._train_model(train_data,
                                       target_column_name,
                                       forced_predictors)))
        else:
            self._forward_selection(train_data, target_column_name,
                                    filterd_predictors,
                                    forced_predictors)

    def _forward_selection(self, train_data: pd.DataFrame,
                           target_column_name: str, predictors: list,
                           forced_predictors: list=[]):
        """Summary

        Parameters
        ----------
        train_data : pd.DataFrame
            Description
        target_column_name : str
            Description
        predictors : list
            Description
        forced_predictors : list, optional
            Description
        """
        current_predictors = []

        max_steps = 1 + min(self.max_predictors,
                            len(predictors) + len(forced_predictors))
        for step in range(1, max_steps):
            if step <= len(forced_predictors):
                # first, we go through forced predictors
                candidate_predictors = list(set(forced_predictors)
                                            .difference(
                                                set(current_predictors)))
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

                self._fitted_models.append(model)

        if not self._fitted_models:
            log.error("No models found in forward selection")

    def _find_next_best_model(self, train_data: pd.DataFrame,
                              target_column_name: str,
                              candidate_predictors: list,
                              current_predictors: list) -> MLModel:
        """Summary

        Parameters
        ----------
        train_data : pd.DataFrame
            Description
        target_column_name : str
            Description
        candidate_predictors : list
            Description
        current_predictors : list
            Description

        Returns
        -------
        MLModel
            Description
        """
        # placeholders
        best_model = None
        best_performance = -1

        for pred in candidate_predictors:

            # train model with additional predictor
            model = self._train_model(train_data, target_column_name,
                                      (current_predictors + [pred]))
            # Evaluate model
            performance = (model
                           .evaluate(train_data[current_predictors + [pred]],
                                     train_data[target_column_name]))

            if (self.pos_only and (not (model.get_coef() >= 0).all())):
                continue

            # check if model is better than current best model
            # and if yes, replace current best!
            if (performance >= best_performance):
                best_performance = performance
                best_model = model

        return best_model

    def _train_model(self, train_data: pd.DataFrame, target_column_name: str,
                     predictors: list) -> MLModel:
        """Summary

        Parameters
        ----------
        train_data : pd.DataFrame
            Description
        target_column_name : str
            Description
        predictors : list
            Description

        Returns
        -------
        MLModel
            Description
        """
        model = MLModel()

        model.fit(train_data[predictors], train_data[target_column_name])

        return model
