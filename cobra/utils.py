"""Cobra utils."""


def clean_predictor_name(predictor_name: str) -> str:
    """
    Clean the predictor name.

    This is done by stripping the redundant suffix (e.g. "_enc" or "_bin") off
    from the end of the predictor name to return a clean version of the predictor
    """
    return (
        predictor_name.replace("_enc", "").replace("_bin", "").replace("_processed", "")
    )
