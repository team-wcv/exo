import ast
from pathlib import Path


def test_pyinstaller_spec_copies_exo_distribution_metadata() -> None:
    spec_path = Path(__file__).parents[1] / "packaging" / "pyinstaller" / "exo.spec"
    tree = ast.parse(spec_path.read_text(encoding="utf-8"), filename=str(spec_path))

    datas = next(
        statement
        for statement in tree.body
        if isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
        and statement.target.id == "DATAS"
    )
    assert isinstance(datas.value, ast.List)
    assert any(
        isinstance(element, ast.Starred)
        and isinstance(element.value, ast.Call)
        and isinstance(element.value.func, ast.Name)
        and element.value.func.id == "copy_metadata"
        and len(element.value.args) == 1
        and isinstance(element.value.args[0], ast.Constant)
        and element.value.args[0].value == "exo"
        for element in datas.value.elts
    )
