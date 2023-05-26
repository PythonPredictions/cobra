"""Quality assessment tasks. Run all quality checks with `inv qa`.
The quality checks are:
- black
- flake8
- pylint
- mypy

"""

from invoke import task
from .colors import colorize, Color

from .system import PTY


# @task
# def black(c_r):
#     """Run code formatter: black."""
#     tmp_str = colorize("\nRunning black...\n", color=Color.HEADER, bold=True)
#     print(f"{tmp_str}")
#     c_r.run(f"black {c_r.project_slug}", pty=PTY)


@task
def flake(c_r):
    """Run style guide enforcement: flake8."""
    tmp_str = colorize("\nRunning flake8...\n", color=Color.HEADER, bold=True)
    print(f"{tmp_str}")
    c_r.run(f"flake8 {c_r.project_slug}", warn=True, pty=PTY)


@task
def pylint(c_r):
    """Run code analysis: pylint."""
    tmp_str = colorize("\nRunning pylint...\n", color=Color.HEADER, bold=True)
    print(f"{tmp_str}")
    c_r.run(f"pylint {c_r.project_slug}", warn=True, pty=PTY)


@task
def mypy(c_r):
    """Run static type checking: mypy."""
    tmp_str = colorize("Running mypy...\n", color=Color.HEADER, bold=True)
    print(f"{tmp_str}")
    c_r.run(f"mypy {c_r.project_slug}", warn=True, pty=PTY)


# @task(post=[black, flake, pylint, mypy], default=True)
@task(post=[flake, pylint, mypy], default=True)
def all(c_r):  # pylint: disable=W0622,W0613 # noqa: F811
    """Run all quality checks."""
