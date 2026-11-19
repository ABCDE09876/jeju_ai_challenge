# CLAUDE.md - 제주의 두 얼굴 (Jeju True-Life & Citizen Lawyer Agent)

「제주 AI·빅데이터 활용 AI Agent 개발 챌린지」(제주테크노파크, 마감 2026-11-20 10:00) 출품작.
관광객·기상·평균의 3대 허수를 벗겨 도민 실생활 결론을 내고, 통계 주장·민원을 감사하는 에이전트.

## 핵심 규칙
- 대용량 CSV/XLSX/SQLite/로그를 통째로 읽지 않는다(.claudeignore). 구조 파악은 스크립트로 시트명·컬럼·자료형·예시 행만 출력.
- 숫자는 로컬 파이썬 도구(SQLite + pandas)만 계산한다. Claude 는 도구 결과의 수치만 인용한다(number_auditor 로 재검증).
- 모든 답변은 세 구획을 분리한다: [원자료로 확인된 수치 사실] / [교란 제거 후 역전 구간] /
  [원자료 컬럼 부재로 인해 단정할 수 없는 한계 및 현장 확인 질문 1개].
- Claude 경로는 claude-opus-5-5 + effort xhigh 고정. 비용 상한(호출 횟수·토큰 축소·저가 모델 대체) 없음.
  max_tokens 는 API 필수값이라 모델 최대치 128000. temperature 미전송(Opus 5.5 미지원).
- 키 미설정·오프라인·인증/네트워크 오류·429·3.5초 무응답·거절 시 화면 오류 없이 로컬 엔진으로 자동 전환.
- API Key 는 .env 에만. 소스·문서·제출물에 하드코딩 금지(scripts/audit_codebase.py 가 검사).
- 이름 규칙: data/info/result/process/handle/manager/util/temp/obj 금지, _list/_dict/_df 접미사 금지,
  함수명은 동사로 시작하는 3단어 이하 snake_case.
- 합성 데이터는 source_type 으로 정직하게 표기(synthetic_seed, calibrated_reference, official_example 등).

## 구조
- src/config.py: 경로·시드(20261120)·Claude 고정 정책
- src/agent/: claude_bridge(Opus 5.5 Tool Use 루프), intent_parser(자모 퍼지 NLU·멀티턴), orchestrator(자동 폴백),
  grounded_synthesizer(근거 묶음·로컬 해설), number_auditor, response_cache, dialogue_memory, prompts
- src/tools/: catalog_resolver(Tool1), confounder_decomposer(Tool2), slice_excavator(Tool3), boundary_simulator(Tool4),
  tool_registry(JSON Schema), resilient_loader, csv_auditor, db_connector, analysis_cubes, jeju_db_builder, synthetic_seed
- src/ui/: components(3단 대시보드), evidence_card(1페이지 HTML), chart_style(한글 글꼴·팔레트); src/app.py(Streamlit)
- data/meta/: catalog_schema.json(공식 엑셀 해독), jeju_gazetteer.json(43개 읍면동·별칭), benchmark_items.json
- data/processed/: jeju_truelife.db(자동 구축), 벤치마크·퍼징 리포트

## 명령어
```bash
pip install -r requirements.txt
python scripts/inspect_official_meta.py      # 공식 엑셀 2종 해독 -> data/meta/catalog_schema.json
python scripts/build_gazetteer.py            # 지명 사전
python scripts/build_jeju_db.py              # 로컬 DB 구축(약 7~40초)
streamlit run src/app.py                     # 앱 실행
python -m pytest -q                          # 단위·폴백·엣지케이스 테스트
python scripts/run_benchmark.py              # 24문항 벤치마크
python scripts/run_fuzz_stress_test.py       # 50문항 퍼징(--live N 으로 실제 API 추가 검증)
python scripts/run_robustness_test.py        # LLM 변형 질문 견고성 시험(--pipeline 8: 전체 답변 경로까지)
python scripts/run_scale_test.py --tag 이름  # 대규모 조합 행동 시험(세트당 1,275만 문항, --per-set 20000 빠른 확인)
python scripts/run_scale_test.py --suite diverse --per-set 35500000 --tag 이름  # 다양성 시험(7,100만 문항, --parts/--part 로 PC 분할)
python scripts/run_scale_test.py --suite mega --per-set 50000000 --tag 이름     # 1억 문항 시험(다양성 + 새 값·새 입력 변형)
python scripts/run_scale_test.py --suite restyle --per-set 30000000 --tag 이름  # 6천만 문항 새 말투 시험(새 말투 24종 문장틀 + 새 입력 변형 10종)
python scripts/audit_codebase.py             # 키·비용상한·이름 규칙 감사
python scripts/generate_proposal_deck.py     # 15쪽 기획서 HTML/PDF/PPTX + 차트 5장
python scripts/package_submission.py         # 제출 ZIP(기본 local-only)
```
