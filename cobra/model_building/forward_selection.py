import logging
log = logging.getLogger(__name__)

import pandas as pd

from cobra.model_building import LogisticRegressionModel as MLModel


class ForwardFeatureSelection:

    """Summary

    Attributes
    ----------
    max_predictors : int
        Description
    model_name : str
        Description
    pos_only : bool
        Description
    """

    def __init__(self, max_predictors: int=50,
                 model_name: str="logistic-regression", pos_only: bool=True):

        self.pos_only = pos_only
        self.max_predictors = max_predictors
        self.model_name = model_name

        self._fitted_models = []

    def fit(self, data: pd.DataFrame, target_column_name: str,
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
        # prep predictor lists
        filterd_predictors = [var for var in predictors
                              if (var not in excluded_predictors
                                  and var not in forced_predictors)]

        # checks on predictor lists and self.max_predictors attr
        if len(forced_predictors) > self.max_predictors:
            raise ValueError("Size or forced_predictors cannot be bigger than "
                             "max_predictors")
        elif len(forced_predictors) == self.max_predictors:
            log.info("Size of forced_predictors equals max_predictors "
                     "only one model will be trained...")
            # train model with all forced_predictors (only)
            (self._fitted_models
             .append(self._train_model(data[data["split"] == "train"],
                                       target_column_name,
                                       forced_predictors)))
        else:
            self._forward_selection(data, target_column_name,
                                    filterd_predictors,
                                    forced_predictors)

    def _forward_selection(self, data: pd.DataFrame, target_column_name: str,
                           predictors: list, forced_predictors: list=[]):
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
        """
        current_predictors = []

        for step in range(1, self.max_predictors + 1):
            if step <= len(forced_predictors):
                # first, we go through forced predictors
                candidate_predictors = list(set(forced_predictors)
                                            .difference(
                                                set(current_predictors)))
            else:
                candidate_predictors = [var for var in predictors
                                        if var not in current_predictors]

            model = self._find_next_best_model(data[data["split"] == "train"],
                                               target_column_name,
                                               candidate_predictors,
                                               current_predictors)
            # if no new model was found, e.g. because there was no model with
            # only positive coefficients, and all forced predictors were
            # already tested (i.e. we are now looping through the other
            # predictors) break out of the loop!
            if (model is None) and (step > len(forced_predictors)):
                break

            if model is not None:
                self._fitted_models.append(model)

        if not self._fitted_models:
            log.error("No models found in forward selection")

    def _find_next_best_model(self, data: pd.DataFrame,
                              target_column_name: str,
                              candidate_predictors: list,
                              current_predictors: list) -> MLModel:
        """Summary

        Parameters
        ----------
        data : pd.DataFrame
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
        best_auc = -1

        for pred in candidate_predictors:

            # train model with additional predictor
            model = self._train_model(data, target_column_name,
                                      (current_predictors + [pred]))
            # Evaluate model
            auc_pred = model.evaluate(data[current_predictors + [pred]],
                                      data[target_column_name])

            if (self.pos_only and (not (model.get_coef() >= 0).all())):
                continue

            # check if model is better than current best model
            # and if yes, replace current best!
            if (auc_pred >= best_auc):
                best_auc = auc_pred
                best_model = model

        return best_model

    def _train_model(self, data: pd.DataFrame, target_column_name: str,
                     predictors: list) -> MLModel:
        """Summary

        Parameters
        ----------
        data : pd.DataFrame
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

        model.fit(data[predictors], data[target_column_name])

        return model
