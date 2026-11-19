"""(개발 도구, 선택) Ollama 로컬 LLM 으로 대규모 시험의 문장틀을 여러 말투로 다시 쓴다.

자리표시자(<P1>, <TIME> 등)는 글자 그대로 남기게 하고, 값은 나중에 scripts/run_scale_test.py 가 끼운다.
그래서 LLM 이 만든 문장도 정답 슬롯이 끼운 값으로 정확히 정해진다. 제출물 실행에는 필요 없다.
2026-11-16 생성 조건: 데스크탑 RTX 5070 Ti 16GB, Ollama 0.34.2, gemma4:12b(개발용)·qwen3:14b(평가용),
temperature 0.9, num_ctx 4096, 추론(think) 끔, 형식 = JSON 스키마, 호출마다 변형 10개.
자리표시자 틀은 (틀, 말투)마다 4번, 자리표시자 없는 문장은 2번 부른다(표본 시드만 다름). 표준 라이브러리만 쓰며
끊겨도 이어서 돈다.
python scripts/generate_question_frames.py --model gemma4:12b --out frames.jsonl [--host http://127.0.0.1:11434]
--suite diverse (2차 다양성 시험, scripts/diverse_suite.py): 질문 유형 57종 x 말투(--styles dev = 개발용 24종,
  all = 36종) x 호출마다 다른 상황(화자·경로·목적·감정). 상황 요소도 --styles dev 이면 평가용 전용 요소를 빼고 고른다.
python scripts/generate_question_frames.py --suite diverse --styles dev --reps 3 --model gemma4:12b --out d_gemma4.jsonl
--suite restyle (6천만 문항 새 말투 시험, scripts/restyle_suite.py): 같은 질문 유형 57종 x 새 말투(--styles dev = 개발용 12종,
  all = 24종) x 호출마다 다른 새 상황. 규칙(시스템 지시)은 diverse 와 같고, 시드는 2000 + 틀 순번으로 정해 이전 생성과 겹치지 않는다.
python scripts/generate_question_frames.py --suite restyle --styles dev --reps 3 --model gemma4:12b --out r_gemma4.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diverse_suite  # noqa: E402
import restyle_suite  # noqa: E402
from scale_suite import PLACEHOLDER, PLACEHOLDER_NOTES, STYLES, load_frames  # noqa: E402

SITUATION_SUITES = {"diverse": (diverse_suite, 1000), "restyle": (restyle_suite, 2000)}  # 시험 정의 모듈, 시드 기준 수

PER_CALL = 10
SYSTEM_SLOTS = (
    "너는 한국어 질문 문장틀 변형기다. 꺾쇠 자리표시자(<P1>, <TIME> 같은 것)가 든 질문을, 뜻은 그대로 두고 지정한 말투로 "
    "여러 번 다시 쓴다.\n"
    "규칙:\n"
    "1. 자리표시자는 꺾쇠까지 글자 그대로, 각각 정확히 한 번씩 쓴다. 자리표시자를 새로 만들거나 바꾸지 않는다.\n"
    "2. 자리표시자에 들어갈 말을 짐작해 쓰지 않고, 자리표시자 앞뒤에 같은 뜻의 말(시간·날씨·장소 등)을 덧붙이지 않는다.\n"
    "3. 자리표시자 바로 뒤에 조사를 붙일 때는 '(이)랑', '은(는)', '이(가)', '을(를)', '과(와)', '(으)로'처럼 두 형태를 함께 쓰고, "
    "원문의 조사 뜻을 바꾸지 않는다(예: '에서'를 '(이)랑'으로 바꾸지 않음).\n"
    "4. 묻는 내용(비교, 언제, 몇 시까지, 사실 여부 등)과 대상, 단위(분·배·년)는 바꾸거나 빼지 않는다. 원문에 없는 장소·시간·"
    "조건·숫자를 넣지 않는다.\n"
    "5. 말투만 바꾸고, 변형마다 한 문장(길어도 짧은 두 문장)으로 서로 다르게 쓴다. 따옴표나 설명을 붙이지 않는다.\n"
    "6. JSON 으로만 답한다."
)
SYSTEM_FIXED = (
    "너는 한국어 질문 변형 생성기다. 주어진 질문과 뜻이 완전히 같은 질문을 지정한 말투로 여러 번 다시 쓴다.\n"
    "규칙:\n"
    "1. 장소(지명), 시간·시각, 날씨, 요일·장날, 대상(어르신·학생 등), 숫자, 연도·계절, 묻는 내용(비교, 언제, 몇 시까지, "
    "사실 여부 등)을 바꾸거나 빼지 않는다.\n"
    "2. 원문에 없는 조건이나 장소를 새로 넣지 않는다.\n"
    "3. 지명은 원문 표기를 그대로 쓴다. typo 말투에서만 지명에 한 글자 정도 오타를 낼 수 있다.\n"
    "4. 변형마다 한 문장(길어도 짧은 두 문장)으로 서로 다르게 쓴다. 따옴표나 설명을 붙이지 않는다.\n"
    "5. JSON 으로만 답한다."
)
SYSTEM_DIVERSE = (
    "너는 한국어 질문 문장틀 변형기다. 꺾쇠 자리표시자(<P1>, <TIME> 같은 것)가 든 질문을, 묻는 뜻은 그대로 두고 지정한 "
    "말투와 상황에 맞게 여러 번 다시 쓴다.\n"
    "규칙:\n"
    "1. 자리표시자는 꺾쇠까지 글자 그대로, 각각 정확히 한 번씩 쓴다. 자리표시자를 새로 만들거나 바꾸지 않는다.\n"
    "2. 자리표시자에 들어갈 말을 짐작해 쓰지 않고, 자리표시자 앞뒤에 같은 뜻의 말을 덧붙이지 않는다.\n"
    "3. 자리표시자 바로 뒤 조사는 '(이)랑', '은(는)', '이(가)', '을(를)', '과(와)', '(으)로'처럼 두 형태를 함께 쓰고, "
    "원문의 조사 뜻을 바꾸지 않는다.\n"
    "4. 묻는 내용(비교, 언제, 몇 시까지, 사실 여부 등)과 원문의 조건·단위(분·배·년·월·%)는 바꾸거나 빼지 않는다.\n"
    "5. 원문에 없는 장소, 시각·시간대, 날씨, 요일·장날, 대상(어르신·학생·아이 등), 숫자, 주장·소문, 주제(버스·매출·쓰레기·"
    "와이파이·전기차·이사·혼잡 등)를 새로 넣지 않는다. 상황은 말투·어휘·문장 구성·인사말에만 반영한다.\n"
    "6. 변형마다 한 문장(길어도 세 문장)으로 서로 다르게 쓴다. 따옴표나 설명을 붙이지 않는다.\n"
    "7. JSON 으로만 답한다."
)
SCHEMA = {"type": "object", "properties": {"variants": {"type": "array", "items": {"type": "string"}}},
          "required": ["variants"]}
WRITE_LOCK = threading.Lock()


def build_prompt(frame: dict, style: str, situation: dict | None = None, suite: str = "diverse") -> str:
    module = SITUATION_SUITES[suite][0]
    notes, styles = (module.PLACEHOLDER_NOTES, module.STYLES) if situation else (PLACEHOLDER_NOTES, STYLES)
    lines = [f"원문 질문: {frame['text']}"]
    names = PLACEHOLDER.findall(frame["text"])
    if names:
        lines.append("자리표시자 뜻:")
        lines += [f"- <{name}>: {notes[name]}" for name in dict.fromkeys(names)]
    if frame.get("context"):
        lines.append(f"이전 질문(문맥일 뿐 다시 쓰지 않는다): {frame['context']}")
        lines.append("원문은 이전 질문에 이어지는 짧은 후속 질문이다. 변형도 이전 질문의 지명이나 내용을 되풀이하지 않고 "
                     "후속 질문 형태를 유지한다.")
    lines.append(f"말투: {style} - {styles[style]}")
    if situation:
        lines.append(f"상황: {module.describe_situation(situation)}(상황은 말투·표현에만 반영)")
    lines.append(f"서로 다른 변형 {PER_CALL}개를 variants 배열로 쓴다.")
    return "\n".join(lines)


def call_model(host: str, model: str, system: str, prompt: str, sample_seed: int) -> tuple[dict, float]:
    payload = {"model": model, "stream": False, "format": SCHEMA, "keep_alive": "30m", "think": False,
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
               "options": {"temperature": 0.9, "num_ctx": 4096, "seed": sample_seed}}
    request = urllib.request.Request(f"{host}/api/chat", data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=900) as response:
        body = json.load(response)
    return body, time.perf_counter() - started


def make_variants(job: tuple, host: str, model: str) -> dict:
    frame, style, rep = job[:3]
    situation = job[3] if len(job) > 3 else None
    suite = job[4] if len(job) > 4 else "diverse"
    if situation:  # 다양성·새 말투 시험: 틀 순번·말투 순번으로 시드를 정한다(앞 시험과 겹치지 않게 큰 수를 더함)
        module, offset = SITUATION_SUITES[suite]
        number = offset + [f["id"] for f in module.load_frames()].index(frame["id"])
        style_number = list(module.STYLES).index(style)
        system = SYSTEM_DIVERSE
    else:
        number = int(frame["id"][1:])
        style_number = list(STYLES).index(style)
        system = SYSTEM_FIXED if frame.get("fixed") else SYSTEM_SLOTS
    last_error = ""
    for attempt in range(3):
        try:
            sample_seed = number * 1000 + style_number * 100 + rep * 10 + attempt
            body, seconds = call_model(host, model, system, build_prompt(frame, style, situation, suite), sample_seed)
            texts = [str(t).strip() for t in json.loads(body["message"]["content"]).get("variants", [])]
            texts = [t for t in texts if t]
            if not texts and attempt < 2:
                last_error = "empty"
                continue
            return {"frame_id": frame["id"], "style": style, "rep": rep, "ok": bool(texts), "attempts": attempt + 1,
                    "seconds": round(seconds, 2), "eval_count": body.get("eval_count"), "texts": texts,
                    **({"situation": situation} if situation else {})}
        except Exception as exc:  # noqa: BLE001 - 생성 실패는 기록하고 다음 호출로 넘어간다
            last_error = f"{type(exc).__name__}: {exc}"
    return {"frame_id": frame["id"], "style": style, "rep": rep, "ok": False, "attempts": 3, "error": last_error,
            "texts": []}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ollama 로컬 LLM 문장틀 변형 생성(개발 도구)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reps-slots", type=int, default=4, help="자리표시자 틀의 (틀, 말투)당 호출 수")
    parser.add_argument("--reps-fixed", type=int, default=2, help="자리표시자 없는 문장의 (문장, 말투)당 호출 수")
    parser.add_argument("--suite", choices=("scale", "diverse", "restyle"), default="scale")
    parser.add_argument("--styles", choices=("dev", "all"), default="dev",
                        help="diverse·restyle: 개발용 말투만 또는 전체(diverse 36종, restyle 24종)")
    parser.add_argument("--reps", type=int, default=3, help="diverse·restyle: (틀, 말투)당 호출 수(호출마다 상황이 다름)")
    args = parser.parse_args()
    if args.suite in SITUATION_SUITES:
        module = SITUATION_SUITES[args.suite][0]
        styles = module.DEV_STYLES if args.styles == "dev" else tuple(module.STYLES)
        frames = module.load_frames()
        jobs = []
        for frame_number, frame in enumerate(frames):
            for style in styles:
                for rep in range(args.reps):
                    call_number = frame_number * 10000 + list(module.STYLES).index(style) * 100 + rep
                    situation = module.pick_situation(call_number, heldout=args.styles == "all")
                    jobs.append((frame, style, rep, situation, args.suite))
    else:
        jobs = [(frame, style, rep) for frame in load_frames() for style in STYLES
                for rep in range(args.reps_fixed if frame.get("fixed") else args.reps_slots)]
    out = Path(args.out)
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row["frame_id"], row["style"], row["rep"]))
    todo = [job for job in jobs if (job[0]["id"], job[1], job[2]) not in done]
    print(f"model {args.model} | calls {len(jobs)} | already done {len(done)} | todo {len(todo)}", flush=True)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool, out.open("a", encoding="utf-8") as sink:
        for finished, row in enumerate(pool.map(lambda job: make_variants(job, args.host, args.model), todo), start=1):
            row["model"] = args.model
            with WRITE_LOCK:
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                sink.flush()
            if finished % 25 == 0 or finished == len(todo):
                print(f"{finished}/{len(todo)} calls, {time.perf_counter() - started:.0f}s elapsed", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
