import os
import sys
import pytest


@pytest.mark.skipif(
    os.environ.get("GA_OS") == "windows-latest",
    reason="Black formatting fails on Windows.",
)
def test_formatting():
    """
    Ensure project adheres to black style
    """
    project_path = os.path.join(os.path.dirname(__file__), "..")
    package_path = os.path.join(project_path, "gap_statistic")
    tests_path = os.path.join(project_path, "tests")
    cmd = [
        sys.executable,
        "-m",
        "black",
        "--check",
        "-v",
        package_path,
        tests_path,
        "--exclude",
        r".*_version.py",
    ]
    exit_code = os.system(" ".join(cmd))
    assert exit_code == 0
