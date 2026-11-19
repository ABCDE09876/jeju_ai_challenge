"""자가 치유 로더·점진적 조건 완화·안전 연산."""
import pandas as pd

from src.tools.resilient_loader import average_safely, divide_safely, load_any_csv, relax_sparse_filter


def test_cp949_decoding():
    raw = "기준일자,읍면동,배출량\n2019-10-01,한림읍,\"1,045.23\"\n".encode("cp949")
    table = load_any_csv(raw)
    assert table.ok and table.encoding == "cp949"
    assert list(table.frame.columns) == ["기준일자", "읍면동", "배출량"]
    assert table.frame["배출량"].iloc[0] == 1045.23


def test_masked_cells():
    table = load_any_csv("a,b\n1,***\n2,5\n3,NULL\n".encode("utf-8"))
    assert table.masked_cells == 1
    assert table.frame["b"].isna().sum() == 2
    assert table.frame["b"].dropna().tolist() == [5.0]


def test_mangled_route():
    table = load_any_csv("bus_route_number,bus_route_name\n02월 01일,2-1(월평)\n455,455(오일장)\n".encode("utf-8"))
    assert table.frame["bus_route_number"].tolist() == ["2-1", "455"]
    assert table.restored_codes == 1


def test_delimiter_sniffing():
    for delimiter in ("\t", ";", "|"):
        table = load_any_csv(f"x{delimiter}y\n1{delimiter}2\n3{delimiter}4\n".encode("euc-kr"))
        assert table.delimiter == delimiter
        assert table.frame.shape == (2, 2)


def test_empty_inputs():
    for raw in (b"", b"   \n  ", bytes(range(256))):
        table = load_any_csv(raw)
        assert isinstance(table.frame, pd.DataFrame)


def test_relax_ladder():
    frame = pd.DataFrame({"hour": [9, 9, 10, 18, 18, 18, 18, 18, 18], "value": [1] * 9})
    stages = [("9시만", lambda f: f["hour"] == 9), ("전체", lambda f: f["hour"] >= 0)]
    picked = relax_sparse_filter(frame, stages, min_rows=5, value_column="value")
    assert picked.relaxed and picked.level == 1 and picked.rows == 9
    assert "자동 확장" in picked.note and "N=2" in picked.note


def test_zero_division():
    assert divide_safely(1, 0) is None
    assert divide_safely(None, 3) is None
    assert average_safely([float("nan"), None, "x"]) is None
    assert average_safely([1, 2, 3]) == 2
