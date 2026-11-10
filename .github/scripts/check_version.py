"""Fail unless every place that states the version agrees.

The version lives in pyproject.toml and in ``hiyal.__version__``. When a tag
name is passed (``python check_version.py v0.1.0``), it must match too.
"""

import re
import sys
import tomllib
from pathlib import Path

root = Path(__file__).resolve().parents[2]
project = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
match = re.search(
    r'^__version__\s*=\s*["\']([^"\']+)["\']',
    (root / "src/hiyal/__init__.py").read_text(),
    re.MULTILINE,
)
package = match.group(1) if match else None

versions = {"pyproject.toml": project, "hiyal.__version__": package}
if len(sys.argv) > 1:
    versions["git tag"] = sys.argv[1].removeprefix("v")

for source, value in versions.items():
    print(f"{source:20} {value}")
if len(set(versions.values())) != 1:
    sys.exit("error: the versions above disagree")
print("versions agree")
