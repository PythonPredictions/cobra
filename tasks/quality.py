"""Quality assessment tasks. Run all quality checks with `inv qa`.
The quality checks are:
- black
- flake8
- pylint
- mypy

"""

from invoke import task
from .colors import colorize

from .system import OperatingSystem, get_current_system

PTY = True
if get_current_system() == OperatingSystem.WINDOWS:
    PTY = False


# @task
# def black(c_r):
#     """Run code formatter: black."""
#     print(colorize("-> Running black..."))
#     c_r.run(f"black {c_r.project_slug}", pty=PTY)


@task
def flake(c_r):
    """Run style guide enforcement: flake8."""
    print(colorize("-> Running flake8..."))
    c_r.run(f"flake8 {c_r.project_slug}", warn=True, pty=PTY)


@task
def pylint(c_r):
    """Run code analysis: pylint."""
    print(colorize("-> Running pylint..."))
    c_r.run(f"pylint {c_r.project_slug}", warn=True, pty=PTY)


@task
def mypy(c_r):
    """Run static type checking: mypy."""
    print(colorize("-> Running mypy..."))
    c_r.run(f"mypy {c_r.project_slug}", warn=True, pty=PTY)


# @task(post=[black, flake, pylint, mypy], default=True)
@task(post=[flake, pylint, mypy], default=True)
def all(c_r):  # pylint: disable=W0622,W0613 # noqa: F811
    """Run all quality checks."""
