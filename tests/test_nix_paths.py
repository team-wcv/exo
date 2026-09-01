import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[1]
SELF_PATH_PATTERN = re.compile(r"inputs\.self\s*\+\s*/(?P<path>[^\s);}\]]+)")


def test_nix_self_paths_exist() -> None:
    missing_paths = [
        f"{nix_file.relative_to(REPOSITORY_ROOT)}: /{match.group('path')}"
        for nix_file in REPOSITORY_ROOT.rglob("*.nix")
        for match in SELF_PATH_PATTERN.finditer(nix_file.read_text(encoding="utf-8"))
        if not (REPOSITORY_ROOT / match.group("path")).exists()
    ]

    assert missing_paths == []
