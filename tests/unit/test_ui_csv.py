"""The CSV logic inside the /ui page, executed rather than eyeballed.

The parser is the one piece of that page where being almost right is silently
wrong: ``split(",")`` looks fine until a review contains a comma inside quotes,
and then every column after it shifts by one for that row only. These tests pull
the real functions out of ``ui.html`` and run them under Node, so the assertions
cover the code that ships, not a copy of it.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

UI = Path(__file__).resolve().parents[2] / "src" / "sentiment" / "serving" / "ui.html"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def _extract(*names: str) -> str:
    """Return the named top-level functions from the page's script block."""
    source = UI.read_text(encoding="utf-8")
    out = []
    for name in names:
        match = re.search(rf"^function {name}\(.*?^}}", source, re.S | re.M)
        assert match, f"function {name} not found in ui.html"
        out.append(match.group(0))
    return "\n".join(out)


def run_js(body: str, *functions: str) -> object:
    """Run a snippet against the page's own functions and return its JSON result."""
    script = _extract(*functions) + "\n" + body
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_quoted_comma_does_not_shift_columns() -> None:
    rows = run_js(
        "console.log(JSON.stringify(parseCSV("
        "'text,label\\n\"dạy hay, nhưng tài liệu kém\",negative\\nngắn gọn,positive')));",
        "parseCSV",
    )
    assert rows == [
        ["text", "label"],
        ["dạy hay, nhưng tài liệu kém", "negative"],
        ["ngắn gọn", "positive"],
    ]


def test_escaped_quotes_and_newlines_inside_a_field() -> None:
    rows = run_js(
        "console.log(JSON.stringify(parseCSV(" '\'text\\n"thầy nói ""rất hay""\\nxuống dòng"\')));',
        "parseCSV",
    )
    assert rows == [["text"], ['thầy nói "rất hay"\nxuống dòng']]


def test_bom_is_stripped_from_the_first_header() -> None:
    """Excel writes a BOM; without stripping it the first column never matches."""
    rows = run_js(
        "console.log(JSON.stringify(parseCSV('\\uFEFFtext,actual\\nxin chào,positive')));",
        "parseCSV",
    )
    assert rows[0][0] == "text"


def test_crlf_line_endings() -> None:
    rows = run_js("console.log(JSON.stringify(parseCSV('a,b\\r\\n1,2\\r\\n')));", "parseCSV")
    assert rows == [["a", "b"], ["1", "2"]]


def test_round_trip_through_the_writer() -> None:
    rows = run_js(
        "console.log(JSON.stringify(parseCSV(toCSV("
        '[["text","label"],["có dấu phẩy, và \\"ngoặc\\"","positive"]]))));',
        "parseCSV",
        "toCSV",
    )
    assert rows == [["text", "label"], ['có dấu phẩy, và "ngoặc"', "positive"]]


def test_text_column_is_guessed_by_shape_not_position() -> None:
    """The free-text column is long and mostly unique; an id column is neither."""
    index = run_js(
        "var header=['id','score','feedback'];"
        "var rows=[['1','5','giáo viên giảng bài rất dễ hiểu và nhiệt tình'],"
        "['2','4','tài liệu sơ sài, cần bổ sung thêm ví dụ thực tế'],"
        "['3','5','phòng học hơi nóng nhưng nội dung thì tốt']];"
        "console.log(JSON.stringify(guessTextColumn(header, rows)));",
        "guessTextColumn",
    )
    assert index == 2


def test_named_text_column_wins_over_a_longer_one() -> None:
    index = run_js(
        "var header=['text','notes'];"
        "var rows=[['dạy hay','ghi chú dài hơn hẳn cột text ở bên trái đây rồi'],"
        "['chán','ghi chú dài hơn hẳn cột text ở bên trái đây nữa nhé']];"
        "console.log(JSON.stringify(guessTextColumn(header, rows)));",
        "guessTextColumn",
    )
    assert index == 0
