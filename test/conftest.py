# content of conftest.py
import os

pytest_plugins = []

current_cwd = os.getcwd()

def pytest_configure(config):
    """
    Allows plugins and conftest files to perform initial configuration.
    This hook is called for every plugin and initial conftest
    file after command line options have been parsed.
    """


def pytest_sessionstart(session):
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.
    """


def pytest_sessionfinish(session, exitstatus):
    """
    Called after whole test run finished, right before
    returning the exit status to the system.
    """
    if "runner" in current_cwd:
        for r, d, f in os.walk(current_cwd):
            for file in f:
                if len([n for n in ["outputs", "packages"] if n in file]) == 0:
                    print(os.path.join(r, file))

def pytest_unconfigure(config):
    """
    called before test process is exited.
    """