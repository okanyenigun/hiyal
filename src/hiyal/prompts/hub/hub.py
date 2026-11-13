"""The prompt hub: every prompt Hiyal uses, by name and version, read only when asked for."""

from functools import cache
from pathlib import Path

# The prompts that ship with Hiyal: library/<name>/<version>.md
LIBRARY = Path(__file__).parent / "library"


class PromptHub:
    """Gives access to prompts by name and version.

    A prompt is a text file at ``<root>/<name>/<version>.md``: ``name`` can
    have slashes to group prompts (``"memory/summarize"``), and every
    version of a prompt is its own file (``v1.md``, ``v2.md``), so versions
    can be tried side by side. A file is read the first time it is asked
    for, and kept afterwards.
    """

    def __init__(self, root: str | Path = LIBRARY) -> None:
        """Serve the prompts under ``root``; by default, the ones that ship with Hiyal."""
        self.root = Path(root)

    def get(self, name: str, version: str) -> str:
        """The text of prompt ``name`` at ``version``, without surrounding whitespace.

        Raises ``ValueError`` if there is no such prompt or version.
        """
        path = self.root / name / f"{version}.md"
        if not path.is_file():
            known = self.versions(name)
            if not known:
                raise ValueError(f"there is no prompt {name!r} in {self.root}")
            raise ValueError(
                f"prompt {name!r} has no version {version!r}; it has {', '.join(known)}"
            )
        return _read(path)

    def versions(self, name: str) -> list[str]:
        """The versions of prompt ``name``, sorted; empty if there is no such prompt."""
        folder = self.root / name
        return (
            sorted(path.stem for path in folder.glob("*.md")) if folder.is_dir() else []
        )


@cache
def _read(path: Path) -> str:
    """A prompt file's text, read once per process."""
    return path.read_text(encoding="utf-8").strip()


# The hub every Hiyal component reads its prompts from.
hub = PromptHub()
