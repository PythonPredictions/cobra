import matplotlib.pyplot as plt

def pytest_configure(config):
    plt.ion()
    pass # your code goes here

def pytest_sessionfinish(session, exitstatus):
    """
    Called after whole test run finished, right before
    returning the exit status to the system.
    """
    plt.close("all")

# other config