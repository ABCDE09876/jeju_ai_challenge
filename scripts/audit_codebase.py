"""최종 자체 감사: 하드코딩된 API Key, 비용 상한 장치, 금지된 상투적 이름 규칙 위반을 찾는다.

검사 항목
  1) 비밀값: 'sk-ant-...' 형태의 실제 키가 소스·문서에 있는지(.env 는 제외 - 제출 ZIP 기본 모드에서 제거됨)
  2) 비용 상한: max_tokens=1024, 일일·세션 호출 상한, 모델 하향 대체 등 요금 방어 코드가 남아 있는지
  3) 이름 규칙: 직접 정의한 변수·함수·인자·클래스 이름에
     - 추상어(data, info, result, process, handle, manager, util, temp, obj) 금지
     - 자료형 접미사(_list, _dict, _df) 금지
     - 함수명은 동사로 시작, 3단어 이하(snake_case)
python scripts/audit_codebase.py  (위반 0건이면 종료코드 0)
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_DIRS = ["src", "scripts", "tests"]
TEXT_GLOBS = ["*.py", "*.md", "*.txt", "*.json", "*.jsonl", "*.toml", "*.example", "*.html"]
BANNED = {"data", "info", "result", "results", "process", "processed_data", "handle", "handler", "manager", "util",
          "utils", "temp", "tmp", "obj"}
BANNED_SUFFIXES = ("_list", "_dict", "_df")
SECRET = re.compile(r"sk-ant-[A-Za-z0-9_\-]{30,}")
BILLING_PATTERNS = [
    (re.compile(r"max_tokens\s*=\s*1024\b"), "max_tokens=1024 상한"),
    (re.compile(r"(daily|session)_(call|request)_?(limit|cap|count)", re.I), "일일/세션 호출 상한"),
    (re.compile(r"MAX_(CALLS|REQUESTS|API_CALLS)", re.I), "호출 횟수 상한 상수"),
    (re.compile(r"claude-3-5-haiku|claude-3-5-sonnet|claude-sonnet-4-20250514"), "저가·구형 모델 대체"),
    (re.compile(r"temperature\s*=\s*0"), "temperature 고정(Opus 5.5 미지원)"),
]
VERBS = set("""
add answer append apply ask assign attach audit average boot build calculate call cast check classify clean clear coerce
collect compact compose compute configure convert count create decide decode decompose describe detect divide draw
emit ensure estimate evaluate excavate explain export extend extract fetch fill filter find format freeze generate get
guess has inject inspect is list load locate log looks make map mark match measure merge name normalize open parse pick
plan probe profile pump raise rank read recall record relax relay remember render repair report request reset resolve
restore run save scan score search serialize shorten show simulate sniff split stamp stream strip style suggest summarize
synthesize tag test tidy update validate warm write jitter stack skip quote main inspect_sheet ask_live audit_sample
""".split())
VERBS.update({"expand", "mask", "stringify", "visit", "divide", "average", "export", "convert", "recall", "assign",
              "quote", "pump", "inject", "audit", "ask", "prepare", "erase"})
EXEMPT_FUNCTIONS = {"main", "get_final_message", "setUp", "tearDown"}
FRAMEWORK_NAMES = re.compile(r"^visit_[A-Za-z]+$")  # ast.NodeVisitor 가 요구하는 메서드 이름 규약


def split_words(name: str) -> list[str]:
    name = name.strip("_")
    parts = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower().split("_")
    return [p for p in parts if p]


class NameCollector(ast.NodeVisitor):
    def __init__(self):
        self.names: list[tuple[str, str, int]] = []

    def add(self, kind: str, name: str, line: int) -> None:
        self.names.append((kind, name, line))

    def visit_FunctionDef(self, node):  # noqa: N802 - ast 방문자 규약
        self.add("function", node.name, node.lineno)
        for arg in node.args.args + node.args.kwonlyargs + node.args.posonlyargs:
            self.add("arg", arg.arg, node.lineno)
        if node.args.vararg:
            self.add("arg", node.args.vararg.arg, node.lineno)
        if node.args.kwarg:
            self.add("arg", node.args.kwarg.arg, node.lineno)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):  # noqa: N802
        self.add("class", node.name, node.lineno)
        self.generic_visit(node)

    def visit_Name(self, node):  # noqa: N802
        if isinstance(node.ctx, ast.Store):
            self.add("variable", node.id, node.lineno)

    def visit_Attribute(self, node):  # noqa: N802
        if isinstance(node.ctx, ast.Store) and isinstance(node.value, ast.Name) and node.value.id == "self":
            self.add("attribute", node.attr, node.lineno)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):  # noqa: N802
        if node.name:
            self.add("variable", node.name, node.lineno)
        self.generic_visit(node)

    def visit_alias(self, node):  # noqa: N802
        if node.asname:
            self.add("import_alias", node.asname, 0)


def audit_names(path: Path) -> list[str]:
    problems = []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    collector = NameCollector()
    collector.visit(tree)
    for kind, name, line in collector.names:
        if (name.startswith("__") and name.endswith("__")) or (kind == "function" and FRAMEWORK_NAMES.match(name)):
            continue
        words = split_words(name)
        if any(word in BANNED for word in words):
            problems.append(f"{path.relative_to(ROOT)}:{line} 금지 단어 '{name}' ({kind})")
        if any(name.lower().endswith(suffix) for suffix in BANNED_SUFFIXES):
            problems.append(f"{path.relative_to(ROOT)}:{line} 자료형 접미사 '{name}' ({kind})")
        if kind == "function" and name not in EXEMPT_FUNCTIONS:
            if words and words[0] not in VERBS:
                problems.append(f"{path.relative_to(ROOT)}:{line} 함수명이 동사로 시작하지 않음 '{name}'")
            if len(words) > 3:
                problems.append(f"{path.relative_to(ROOT)}:{line} 함수명 3단어 초과 '{name}'")
    return problems


def scan_text(path: Path) -> list[str]:
    problems = []
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return problems
    for match in SECRET.finditer(text):
        problems.append(f"{path.relative_to(ROOT)} 실제 API Key 형태 문자열: {match.group(0)[:12]}...")
    if path.suffix == ".py" and path.name != "audit_codebase.py":
        for pattern, label in BILLING_PATTERNS:
            for match in pattern.finditer(text):
                line = text[: match.start()].count("\n") + 1
                problems.append(f"{path.relative_to(ROOT)}:{line} 비용 상한·모델 대체 흔적({label})")
    return problems


def run_audit() -> dict:
    name_problems, secret_problems = [], []
    for folder in CODE_DIRS:
        for path in sorted((ROOT / folder).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            name_problems += audit_names(path)
    text_files = set()
    for pattern in TEXT_GLOBS:
        text_files.update(p for p in ROOT.rglob(pattern) if p.is_file())
    for path in sorted(text_files):
        parts = set(path.relative_to(ROOT).parts)
        if parts & {".venv", "venv", "node_modules", "__pycache__", "dist", ".git", "processed"} or path.name == ".env":
            continue
        secret_problems += scan_text(path)
    return {"naming": name_problems, "secrets_and_billing": secret_problems}


def main() -> int:
    report = run_audit()
    total = len(report["naming"]) + len(report["secrets_and_billing"])
    for section, problems in report.items():
        print(f"[{section}] {len(problems)}건")
        for problem in problems:
            print("  -", problem)
    print("감사 통과" if total == 0 else f"위반 {total}건")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
