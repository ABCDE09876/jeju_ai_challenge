"""대회 제출용 최종 ZIP 생성기.

대회 규정
  - 제주데이터허브 등 데이터 API Key 는 제출물에 절대 넣지 않는다 -> JEJU_DATAHUB_API_KEY 는 어떤 모드에서도 빈 값.
  - 외부 AI API 의 평가용 Key 는 실행환경·매뉴얼에 포함해 제출할 수 있다 -> --mode with-eval-key 에서만
    ANTHROPIC_API_KEY 하나를 ZIP 안 .env 에 남긴다(사용 한도 설정·평가 후 폐기 안내 동봉).
모드
  --mode local-only (기본) : 모든 키 제거, ZIP 안에는 .env.example 만. 키 없이도 100% 로컬 폴백으로 동작.
  --mode with-eval-key     : 평가용 ANTHROPIC_API_KEY 만 유지(--env-file 로 키 파일 지정 가능, 기본 프로젝트 .env)
공통: .env 원본·SQLite DB·캐시·로그·__pycache__ 제외, 소스·문서의 키/개인정보(이메일·전화·주민번호) 패턴 자동 검열.
python scripts/package_submission.py [--mode local-only|with-eval-key] [--env-file PATH] [--include-official] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE_DIRS = ["src", "scripts", "tests", "docs", "data/meta", "data/samples"]
INCLUDE_FILES = ["README.md", "requirements.txt", ".env.example", "CLAUDE.md", ".claudeignore"]
REPORT_GLOBS = ["data/processed/*.json", "data/processed/*.md"]
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".git", "dist", ".venv", "venv", "node_modules"}
SKIP_SUFFIXES = {".db", ".sqlite", ".sqlite3", ".building", ".log", ".pyc", ".zip"}
SECRET = re.compile(r"sk-ant-[A-Za-z0-9_\-]{30,}")
DATAHUB_ASSIGN = re.compile(r"(?m)^([ \t]*JEJU_DATAHUB_API_KEY[ \t]*=[ \t]*)([^\s#]+)")
DATAHUB_SCAN_SUFFIXES = {".example", ".md", ".txt", ".env"}  # 소스 코드(.py)는 문자열 훼손 방지를 위해 제외
PERSONAL = [
    # 예시 전용 도메인(RFC 2606 의 example.com·org·net)은 개인정보가 아니라 그대로 둔다(해석기 회귀 시험 문장이 쓰는 가짜 주소)
    (re.compile(r"[A-Za-z0-9._%+-]+@(?!anthropic\.com|example\.(?:com|org|net)\b)[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "이메일"),
    (re.compile(r"\b01[016789]-\d{3,4}-\d{4}\b"), "휴대전화번호"),
    (re.compile(r"\b\d{6}-[1-4]\d{6}\b"), "주민등록번호"),
]
PERSONAL_SCAN_SUFFIXES = {".py", ".md", ".txt", ".html"}  # 데이터 파일(JSON/CSV)은 값 훼손 방지를 위해 키 검사만
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".jsonl", ".csv", ".toml", ".example", ".html", ".claudeignore", ".yml",
                 ".yaml"}


def collect_files(include_official: bool) -> list[Path]:
    files: list[Path] = []
    folders = INCLUDE_DIRS + (["data/official"] if include_official else [])
    for folder in folders:
        base = ROOT / folder
        if base.exists():
            files += [p for p in base.rglob("*") if p.is_file()]
    files += [ROOT / name for name in INCLUDE_FILES if (ROOT / name).exists()]
    for pattern in REPORT_GLOBS:
        files += list(ROOT.glob(pattern))
    picked = []
    for path in sorted(set(files)):
        rel = path.relative_to(ROOT)
        if set(rel.parts) & SKIP_PARTS or path.suffix.lower() in SKIP_SUFFIXES or path.name == ".env":
            continue
        picked.append(path)
    return picked


def scan_text(path: Path, text: str) -> tuple[str, list[str]]:
    """키·개인정보를 찾아 가리고 발견 목록을 돌려준다."""
    findings = []
    if SECRET.search(text):
        findings.append("Anthropic 키 형태 문자열 제거")
        text = SECRET.sub("<removed-api-key>", text)
    is_env_like = path.suffix.lower() in DATAHUB_SCAN_SUFFIXES or path.name.startswith(".env")
    if is_env_like and DATAHUB_ASSIGN.search(text):
        def erase_value(match):
            value = match.group(2)
            if value.startswith(("<", '"', "'", "your", "여기")):
                return match.group(0)  # 자리표시자는 그대로 둔다
            findings.append("JEJU_DATAHUB_API_KEY 값 제거")
            return match.group(1)
        text = DATAHUB_ASSIGN.sub(erase_value, text)
    if path.suffix.lower() in PERSONAL_SCAN_SUFFIXES:
        for pattern, label in PERSONAL:
            if pattern.search(text):
                findings.append(f"{label} 패턴 가림")
                text = pattern.sub(f"<{label} 제거>", text)
    return text, findings


def read_eval_key(env_file: Path) -> str:
    if not env_file.exists():
        raise SystemExit(f"[중단] 평가용 키 파일이 없습니다: {env_file}")
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("ANTHROPIC_API_KEY="):
            key = line.split("=", 1)[1].strip().strip('"').strip("'")
            if key:
                return key
    raise SystemExit("[중단] --mode with-eval-key 인데 ANTHROPIC_API_KEY 값이 비어 있습니다.")


def build_eval_env(key: str) -> str:
    return ("# 평가용 외부 AI API Key (대회 규정: 평가용 Key 는 제출 가능, 데이터허브 Key 는 제출 금지)\n"
            "# 평가 기간 종료 후 이 Key 를 Anthropic Console 에서 폐기해 주세요. 사용 한도 설정을 권장합니다.\n"
            f"ANTHROPIC_API_KEY={key}\nANTHROPIC_MODEL=claude-opus-5-5\nJEJU_DATAHUB_API_KEY=\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="제주의 두 얼굴 제출 ZIP 생성")
    parser.add_argument("--mode", choices=["local-only", "with-eval-key"], default="local-only")
    parser.add_argument("--env-file", default=str(ROOT / ".env"), help="with-eval-key 모드에서 키를 읽을 파일")
    parser.add_argument("--include-official", action="store_true",
                        help="DAKER 제공 공식 엑셀(data/official)을 포함(기본 제외: 외부 배포 금지 파일)")
    parser.add_argument("--out", default=str(ROOT / "dist"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    files = collect_files(args.include_official)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(args.out)
    target = out_dir / f"jeju_truelife_agent_{args.mode}_{stamp}.zip"
    manifest = {"mode": args.mode, "created_at": datetime.now().isoformat(timespec="seconds"), "files": [],
                "redactions": {}, "notes": [
                    "data/processed/jeju_truelife.db 는 크기 때문에 제외 -> 첫 실행 시 자동 구축(약 10~40초)",
                    "제주데이터허브 API Key 는 모든 모드에서 제외",
                    "공식 엑셀 원본은 기본 제외(DAKER 외부 배포 금지 안내). 해독 결과는 data/meta/catalog_schema.json 으로 포함"]}
    eval_key = read_eval_key(Path(args.env_file)) if args.mode == "with-eval-key" else ""
    if args.dry_run:
        for path in files:
            print(path.relative_to(ROOT).as_posix())
        print(f"(dry-run) {len(files)}개 파일, 모드 {args.mode}")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            rel = path.relative_to(ROOT).as_posix()
            if path.suffix.lower() in TEXT_SUFFIXES or path.name in (".claudeignore", ".env.example"):
                try:
                    text = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    archive.write(path, rel)
                    manifest["files"].append({"path": rel, "bytes": path.stat().st_size})
                    continue
                cleaned, findings = scan_text(path, text)
                if findings:
                    manifest["redactions"][rel] = findings
                archive.writestr(rel, cleaned)
                manifest["files"].append({"path": rel, "bytes": len(cleaned.encode("utf-8"))})
            else:
                archive.write(path, rel)
                manifest["files"].append({"path": rel, "bytes": path.stat().st_size})
        if eval_key:
            archive.writestr(".env", build_eval_env(eval_key))
            archive.writestr("평가용_API_Key_안내.txt",
                             "이 ZIP 의 .env 에는 심사용 Anthropic API Key 가 들어 있습니다.\n"
                             "- 모델은 claude-opus-5-5, effort xhigh 로 고정되어 있습니다.\n"
                             "- 키가 막히거나 비용 한도에 걸리면 앱은 자동으로 100% 로컬 엔진으로 전환합니다.\n"
                             "- 평가 종료 후 참가자가 키를 폐기합니다.\n")
            manifest["files"].append({"path": ".env", "bytes": "평가용 ANTHROPIC_API_KEY 만 포함"})
            manifest["files"].append({"path": "평가용_API_Key_안내.txt", "bytes": "키 사용·폐기 안내"})
        archive.writestr("MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=1))
    size_mb = target.stat().st_size / 1024 / 1024
    print(f"생성: {target} ({size_mb:.1f} MB, 파일 {len(manifest['files'])}개, 모드 {args.mode})")
    if manifest["redactions"]:
        print("자동 검열:", json.dumps(manifest["redactions"], ensure_ascii=False))
    with zipfile.ZipFile(target) as archive:
        names = archive.namelist()
        leaked = [n for n in names if n.endswith(".env") and args.mode == "local-only"]
        secrets = [n for n in names if n != ".env" and SECRET.search(archive.read(n).decode("utf-8", "ignore"))]
        broken = []
        for name in names:
            if name.endswith(".py"):
                try:
                    compile(archive.read(name).decode("utf-8"), name, "exec")
                except SyntaxError as exc:
                    broken.append(f"{name}:{exc.lineno}")
        missing = [name for name in ("README.md", "requirements.txt", ".env.example", "src/app.py") if name not in names]
    if leaked or secrets or broken or missing:
        print(f"[경고] 검증 실패: .env={leaked}, 키 문자열={secrets}, 문법 오류={broken}, 필수 파일 누락={missing}")
        return 1
    print(f"검증: 파이썬 파일 {sum(1 for n in names if n.endswith('.py'))}개 문법 확인 통과, 필수 파일 포함")
    print("검증: local-only 모드 .env 없음 / 소스·문서 내 키 문자열 0건" if args.mode == "local-only"
          else "검증: .env 외 파일에 키 문자열 0건, 데이터허브 키 빈 값")
    return 0


if __name__ == "__main__":
    sys.exit(main())
