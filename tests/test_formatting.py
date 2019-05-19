import os

import pytest


@pytest.mark.skipif(
    os.environ.get("AGENT_OS") == "Windows_NT",
    reason="Black formatting fails on Azure Windows builds.",
)
def test_formatting():
    """
    Ensure project adheres to black style
    """
    import black
    from click.testing import CliRunner

    print(os.environ)

    proj_path = os.path.join(os.path.dirname(__file__), "..", "gap_statistic")
    tests_path = os.path.join(os.path.dirname(__file__), "..", "tests")
    setuppy = os.path.join(os.path.dirname(__file__), "..", "setup.py")

    runner = CliRunner()
    resp = runner.invoke(black.main, ["--check", "-v", proj_path, tests_path, setuppy])
    assert resp.exit_code == 0, "Would still reformat one or more files:\n{}".format(
        resp.output
    )
