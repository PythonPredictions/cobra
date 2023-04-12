"""Test tasks."""

from invoke import task
from .colors import colorize
from .system import (
    OperatingSystem,
    get_current_system,
    COV_DOC_BUILD_DIR,
)

SYSTEM = get_current_system()

PTY = False
if SYSTEM in [OperatingSystem.LINUX, OperatingSystem.MAC]:
    from .system import COV_SCREEN_NAME
    PTY = True


@task(help={"verbose": "Run tests verbose."})
def run(c_r, verbose=True):
    """Run test suite."""
    # print('I am run(c) and I was called!')
    # print('I am c!\n', c_r, '\n', type(c), '\n', c_r.__dict__)  # <-- c_r?
    if verbose:
        c_r.run(
            f"pytest -v -W ignore::UserWarning \
            --cov={c_r.project_slug} --cov-report=term:skip-covered \
            --cov-report=html --cov-report=html:{COV_DOC_BUILD_DIR}",
            pty=PTY,
        )
    else:
        c_r.run(
            f"pytest -W ignore::UserWarning \
            --cov={c_r.project_slug} --cov-report=term:skip-covered \
            --cov-report=html --cov-report=html:{COV_DOC_BUILD_DIR}",
            pty=PTY,
        )


@task
def coverage(c_r):
    """Start coverage report webserver."""
    COV_PORT = c_r.start_port + 2

    if SYSTEM in [OperatingSystem.LINUX, OperatingSystem.MAC]:
        _command = f"screen -d -S {COV_SCREEN_NAME} \
            -m python -m http.server --bind localhost \
            --directory {COV_DOC_BUILD_DIR} {COV_PORT}"
    elif SYSTEM == OperatingSystem.WINDOWS:
        _command = f"wt -d . python -m http.server --bind localhost --directory {COV_DOC_BUILD_DIR} {COV_PORT}"  # noqa: E501 # pylint: disable=line-too-long
    else:
        raise ValueError(f"System {SYSTEM} is not supported")

    c_r.run(_command)

    url = f"http://localhost:{COV_PORT}"

    print("Coverage server hosted in background:\n")
    print(f"-> {colorize(url, underline=True)}\n")
    print(f"Stop server: {colorize('inv test.stop')}\n")


@task
def stop(c_r):
    """Stop coverage report webserver."""

    if SYSTEM in [OperatingSystem.LINUX, OperatingSystem.MAC]:
        result = c_r.run(
            f"screen -ls {COV_SCREEN_NAME}", warn=True, hide="both"
        )
        if "No Sockets" in result.stdout:
            return
        screens = result.stdout.splitlines()[1:-1]
        for scr in screens:
            name = scr.split("\t")[1]
            c_r.run(f"screen -S {name} -X quit")
    elif SYSTEM == OperatingSystem.WINDOWS:
        print(
            "Coverage server is not attached to this process. "
            "Close windows terminal instance instead"
        )
        return
    else:
        raise ValueError(f"System {SYSTEM} is not supported")


@task(post=[stop, run, coverage], default=True)
def all(c_r):  # pylint: disable=W0622,W0613 # noqa: F811
    """Run all tests and start coverage report webserver."""
    # print('I am all(c) and I was called!')
