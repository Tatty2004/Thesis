"""The layering rules from CLAUDE.md, checked on the source.

core/ is game-agnostic: it imports no layer and names no game. The layers (toygames/,
holdem/) build on core/ only and never import each other. experiments/ may use
everything. OpenSpiel appears only in tests/.
"""
import re
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1] / "bombpot"
LAYERS = ["toygames", "holdem"]
SPECIFIC = re.compile(r"\b(kuhn|leduc|board_leduc|holdem|Kuhn|Leduc|BoardLeduc|Holdem|make_game|GAMES)\b")


def imports(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines()
            if line.lstrip().startswith(("import ", "from "))]


def test_core_imports_no_layer_and_names_no_game():
    for path in (PKG / "core").rglob("*.py"):
        for line in imports(path):
            if "bombpot" in line:
                assert line.startswith(("from bombpot.core", "import bombpot.core")), f"{path.name}: {line}"
        assert not SPECIFIC.search(path.read_text()), f"{path.name} names a specific game"


@pytest.mark.parametrize("layer", LAYERS)
def test_layers_build_on_core_only(layer):
    for path in (PKG / layer).rglob("*.py"):
        for line in imports(path):
            if "bombpot" in line:
                assert re.match(rf"(from|import) bombpot\.(core|{layer})\b", line), f"{layer}/{path.name}: {line}"


def test_open_spiel_only_in_tests():
    for path in PKG.rglob("*.py"):
        assert "pyspiel" not in path.read_text() and "open_spiel" not in path.read_text(), path.name
