import logging


def test_log_tutorial(caplog):
    from cobra.utils import log_tutorial

    with caplog.at_level(logging.DEBUG):
        log_tutorial()
    assert (
        "https://github.com/PythonPredictions/cobra/tree/master/tutorials"
        in caplog.text
    )
