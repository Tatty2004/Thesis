"""The layering rules from CLAUDE.md, checked on the source."""
import re
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1] / "toygames"
GAME_AGNOSTIC = ["solvers", "eval", "abstraction"]
SPECIFIC = re.compile(r"\b(kuhn|leduc|board_leduc|Kuhn|Leduc|BoardLeduc|make_game|GAMES)\b")


@pytest.mark.parametrize("package", GAME_AGNOSTIC)
def test_game_agnostic_layers_reference_no_specific_game(package):
    for path in (PKG / package).glob("*.py"):
        src = path.read_text()
        for line in src.splitlines():
            if line.lstrip().startswith(("import ", "from ")) and "toygames.games" in line:
                assert "toygames.games.base" in line, f"{path.name}: {line.strip()}"
        assert not SPECIFIC.search(src), f"{path.name} names a specific game"


def test_open_spiel_only_in_tests():
    for path in PKG.rglob("*.py"):
        assert "pyspiel" not in path.read_text() and "open_spiel" not in path.read_text(), path.name
