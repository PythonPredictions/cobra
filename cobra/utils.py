import logging

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# logger.addHandler(logging.Handler())


def clean_predictor_name(predictor_name: str) -> str:
    """Strip the redundant suffix (e.g. "_enc" or "_bin") off from the end
    of the predictor name to return a clean version of the predictor
    """
    return (
        predictor_name.replace("_enc", "").replace("_bin", "").replace("_processed", "")
    )


def log_tutorial() -> None:
    logging.info(
        """
    Hi, welcome to Cobra!
    You can find some tutorials that explain the functioning of cobra on the PythonPredictions GitHub:
    https://github.com/PythonPredictions/cobra/tree/master/tutorials
        """
    )
