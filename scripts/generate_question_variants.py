"""(개발 도구, 선택) Ollama 로컬 LLM 으로 기준 질문마다 7가지 말투 변형을 만든다.

제출물 실행에는 필요 없다. 견고성 시험 세트(data/meta/robustness_variants_*.jsonl)를 만든 절차를 기록하려고 넣었다.
2026-11-16 생성 조건: 데스크탑 RTX 5070 Ti 16GB, Ollama 0.34.2, gemma4:12b(개발용)·qwen3:14b(평가용),
temperature 0.8, num_ctx 4096, 추론(think) 끔, 표본 시드 = 문항 번호 x 10 + 재시도 번호, 형식 = JSON 스키마.
표준 라이브러리만 쓰며, 같은 출력 파일로 다시 실행하면 끝난 문항은 건너뛴다.
python scripts/generate_question_variants.py --model gemma4:12b --out variants.jsonl [--host http://127.0.0.1:11434]
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STYLES = {
    "casual": "친구에게 묻는 반말 구어체",
    "typo": "급하게 친 듯한 문장. 자판 오타·맞춤법 오류·띄어쓰기 오류를 반드시 2개 이상 넣는다(지명도 한 글자 정도는 틀릴 수 있음)",
    "jeju": "제주 사투리 어미와 표현을 섞은 말투(예: ~우꽈?, ~수과?, ~쿠과?, ~마씸, 하영=많이, 게메=글쎄). 지명과 숫자는 표준어 그대로",
    "elderly": "나이 든 어르신이 묻는 듯한 길고 공손한 말투",
    "short": "검색창에 치듯 아주 짧게 줄인 질문",
    "reorder": "조건을 문장 앞으로 옮기는 등 어순을 바꾼 표준어 질문",
    "mixed": "영어 단어를 몇 개 섞은 질문(지명과 숫자는 한국어 그대로)",
}
SYSTEM = (
    "너는 한국어 질문 변형 생성기다. 주어진 원문 질문과 뜻이 완전히 같은 질문을 지정된 말투마다 하나씩 다시 쓴다.\n"
    "규칙:\n"
    "1. 장소(지명), 시간·시각, 날씨, 요일·장날, 대상(어르신·학생 등), 숫자, 연도·계절, 묻는 내용(비교, 언제, 몇 시까지, "
    "사실 여부 등)을 바꾸거나 빼지 않는다.\n"
    "2. 원문에 없는 조건이나 장소를 새로 넣지 않는다.\n"
    "3. 지명은 원문 표기를 그대로 쓴다. typo 말투에서만 지명에 한 글자 정도 오타를 낼 수 있다.\n"
    "4. 각 말투마다 한 문장(길어도 짧은 두 문장)만 쓰고, 따옴표나 설명을 붙이지 않는다.\n"
    "5. JSON 으로만 답한다."
)
SCHEMA = {
    "type": "object",
    "properties": {"variants": {"type": "array", "items": {
        "type": "object",
        "properties": {"style": {"type": "string", "enum": list(STYLES)}, "text": {"type": "string"}},
        "required": ["style", "text"]}}},
    "required": ["variants"],
}
WRITE_LOCK = threading.Lock()


def build_prompt(seed: dict) -> str:
    lines = [f"원문 질문: {seed['text']}"]
    if seed.get("context"):
        lines.append(f"이전 질문(문맥일 뿐 변형하지 않는다): {seed['context']}")
        lines.append("원문은 이전 질문에 이어지는 짧은 후속 질문이다. 변형도 이전 질문 내용을 되풀이하지 말고 후속 질문 형태를 유지한다.")
    lines.append("말투 목록(이 순서대로 하나씩):")
    lines += [f"- {name}: {desc}" for name, desc in STYLES.items()]
    return "\n".join(lines)


def call_model(host: str, model: str, prompt: str, sample_seed: int) -> tuple[dict, float]:
    payload = {"model": model, "stream": False, "format": SCHEMA, "keep_alive": "30m", "think": False,
               "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
               "options": {"temperature": 0.8, "num_ctx": 4096, "seed": sample_seed}}
    request = urllib.request.Request(f"{host}/api/chat", data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        body = json.load(response)
    return body, time.perf_counter() - started


def make_variants(seed: dict, host: str, model: str) -> dict:
    number = int(seed["id"][1:])
    last_error = ""
    for attempt in range(3):
        try:
            body, seconds = call_model(host, model, build_prompt(seed), number * 10 + attempt)
            parsed = json.loads(body["message"]["content"])
            picked = {}
            for row in parsed.get("variants", []):
                text = str(row.get("text", "")).strip()
                if row.get("style") in STYLES and text and row["style"] not in picked:
                    picked[row["style"]] = text
            missing = [s for s in STYLES if s not in picked]
            if missing and attempt < 2:
                last_error = f"missing {missing}"
                continue
            return {"seed_id": seed["id"], "ok": not missing, "missing": missing, "attempts": attempt + 1,
                    "seconds": round(seconds, 2), "eval_count": body.get("eval_count"), "variants": picked}
        except Exception as exc:  # noqa: BLE001 - 생성 실패는 기록하고 다음 문항으로 넘어간다
            last_error = f"{type(exc).__name__}: {exc}"
    return {"seed_id": seed["id"], "ok": False, "missing": list(STYLES), "attempts": 3, "error": last_error,
            "variants": {}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ollama 로컬 LLM 질문 변형 생성(개발 도구)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--seeds", default=str(ROOT / "data" / "meta" / "robustness_seeds.jsonl"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    seeds = [json.loads(line) for line in Path(args.seeds).read_text(encoding="utf-8").splitlines() if line.strip()]
    out = Path(args.out)
    done = set()
    if out.exists():
        done = {json.loads(line)["seed_id"] for line in out.read_text(encoding="utf-8").splitlines() if line.strip()}
    todo = [s for s in seeds if s["id"] not in done]
    print(f"model {args.model} | seeds {len(seeds)} | already done {len(done)} | todo {len(todo)}", flush=True)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool, out.open("a", encoding="utf-8") as sink:
        for finished, row in enumerate(pool.map(lambda s: make_variants(s, args.host, args.model), todo), start=1):
            row["model"] = args.model
            with WRITE_LOCK:
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                sink.flush()
            if finished % 20 == 0 or finished == len(todo):
                print(f"{finished}/{len(todo)} seeds, {time.perf_counter() - started:.0f}s elapsed", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
