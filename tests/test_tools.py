"""4대 교차검증 도구: 공식 스키마 일치, 역전·은폐 구간·집계 단절·경계 검출, JSON 무결성."""
import json

from src.tools.boundary_simulator import simulate_reversal_boundary
from src.tools.catalog_resolver import resolve_spatial_catalog
from src.tools.confounder_decomposer import decompose_confounders
from src.tools.slice_excavator import excavate_masked_slice
from src.tools.tool_registry import list_claude_tools, run_tool


def test_schema_alignment():
    output = resolve_spatial_catalog(["노형동"], "commute")
    checks = {d["table"]: d["schema_check"] for d in output["datasets"] if d["schema_check"].get("total")}
    assert checks["floating_pop_hourly"]["matched"] == checks["floating_pop_hourly"]["total"] == 11
    assert checks["bus_stop_hourly"]["matched"] == checks["bus_stop_hourly"]["total"] == 9
    assert all(p["source_type"] for p in output["provenance"])


def test_reversal_detection():
    output = decompose_confounders(["애월읍", "아라동"], "living")
    assert output["verdict"] == "reversal"
    assert output["naive_leader"] == "애월읍" and output["final_leader"] == "아라동"


def test_masked_slice():
    output = excavate_masked_slice(["오일장"], "crowd")
    top = output["slices"][0]
    assert output["verdict"] == "masked_overload"
    assert "장날" in top["day_group"] and set(top["hours"]) & {9, 10, 11, 12, 13}


def test_structural_break():
    output = excavate_masked_slice(["한림읍"], "waste", claim="한림읍 음식물쓰레기가 58.69배 늘었다. 관광객 때문이다.")
    assert any(b["month"] == "201911" and b["registered_change"] for b in output["breaks"])
    assert output["verdict"] == "unmeasured_cause"
    assert abs(output["hallim_reference"]["ratio"] - 58.69) < 0.005


def test_boundary_hour():
    output = simulate_reversal_boundary(["아라동", "삼화지구"], "return_hour")
    assert output["boundary"]["boundary"] == 20
    assert len(output["followups"]) == 3 and output["field_question"]


def test_json_clean():
    for name, args in [("resolve_spatial_catalog", {"areas": ["애월"], "topic": "living"}),
                       ("decompose_confounders", {"areas": ["우도면", "추자면"], "hour_from": 3, "hour_to": 3,
                                                  "weather": "rain", "day_type": "market"}),
                       ("excavate_masked_slice", {"areas": ["삼화지구"], "topic": "commute"}),
                       ("simulate_reversal_boundary", {"areas": ["오일장"], "variable": "visit_hour"})]:
        output = run_tool(name, args)
        assert output["ok"], output
        json.dumps(output, ensure_ascii=False, allow_nan=False)


def test_invalid_input():
    output = run_tool("decompose_confounders", {"areas": 5})
    assert output["ok"] is False and "error" in output
    output = run_tool("unknown_tool", {})
    assert output["ok"] is False


def test_claude_schemas():
    tools = list_claude_tools()
    assert [t["name"] for t in tools] == ["resolve_spatial_catalog", "decompose_confounders", "excavate_masked_slice",
                                          "simulate_reversal_boundary"]
    assert all(t["eager_input_streaming"] and t["input_schema"]["type"] == "object" for t in tools)
