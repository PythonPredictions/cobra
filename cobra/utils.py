def clean_predictor_name(predictor: str) -> str:
    """Strip-off redundant suffix (e.g. "_enc" or "_bin") from the predictor
    name to return a clean version of the predictor

    Args:
        predictor (str): Description

    Returns:
        str: Description
    """
    return (predictor.replace("_enc", "").replace("_bin", "")
            .replace("_processed", ""))
