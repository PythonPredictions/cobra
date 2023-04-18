"""Test tasks."""

from invoke import task
from .colors import colorize, Color
from .system import (
    OperatingSystem,
    get_current_system,
    COV_DOC_BUILD_DIR,
    COV_SCREEN_NAME,
    PTY,
)

SYSTEM = get_current_system()


@task(help={"verbose": "Run tests verbose."})
def run(c_r, verbose=False):
    """Run test suite."""
    if verbose:
        c_r.run(
            f"pytest -v -W ignore::UserWarning "
            f"--cov={c_r.project_slug} --cov-report=term:skip-covered "
            f"--cov-report=html --cov-report=html:{COV_DOC_BUILD_DIR}",
            pty=PTY,
        )
    else:
        c_r.run(
            f"pytest -W ignore::UserWarning "
            f"--cov={c_r.project_slug} --cov-report=term:skip-covered "
            f"--cov-report=html --cov-report=html:{COV_DOC_BUILD_DIR}",
            pty=PTY,
        )


@task
def coverage(c_r):
    """Start coverage report webserver."""
    COV_PORT = c_r.start_port + 2

    if SYSTEM in [OperatingSystem.LINUX, OperatingSystem.MAC]:
        _command = (
            f"screen -d -S {COV_SCREEN_NAME} "
            "-m python -m http.server --bind localhost "
            f"--directory {COV_DOC_BUILD_DIR} {COV_PORT}"
        )
    elif SYSTEM == OperatingSystem.WINDOWS:
        _command = (
            f"wt -d . python -m http.server --bind localhost "
            f"--directory {COV_DOC_BUILD_DIR} {COV_PORT}"
        )
    else:
        raise ValueError(f"System {SYSTEM} is not supported")
    tmp_str = colorize(
        "Starting coverage server...",
        color=Color.HEADER,
        bold=True
    )
    print(f"{tmp_str}")
    c_r.run(_command)
    print(f">>> {colorize(_command, color=Color.OKBLUE)}\n")

    url = f"http://localhost:{COV_PORT}"

    print("Coverage server hosted in background:\n")
    print(f"--> {colorize(url, underline=True)}\n")
    print(f"Stop server: {colorize('inv test.stop')}\n")


@task
def stop(c_r):
    """Stop coverage report webserver."""
    COV_PORT = c_r.start_port + 2

    if SYSTEM in [OperatingSystem.LINUX, OperatingSystem.MAC]:
        result = c_r.run(
            f"screen -ls {COV_SCREEN_NAME}", warn=True, hide="both"
        )
        if "No Sockets" in result.stdout:
            return
        tmp_str = colorize(
            "Stopping coverage server...",
            color=Color.HEADER,
            bold=True
        )
        print(tmp_str)
        _command = f"kill $(lsof -ti:{COV_PORT})"
        print(f">>> {colorize(_command, color=Color.OKBLUE)}\n")
        c_r.run(_command)

    elif SYSTEM == OperatingSystem.WINDOWS:
        print(
            colorize(
                "Coverage server is not attached to this process. "
                "Close windows terminal instance instead",
                color=Color.WARNING,
            )
        )
        return
    else:
        raise ValueError(f"System {SYSTEM} is not supported")


@task(post=[stop, run, coverage], default=True)
def all(c_r):  # pylint: disable=W0622,W0613 # noqa: F811
    """Run all tests and start coverage report webserver."""
