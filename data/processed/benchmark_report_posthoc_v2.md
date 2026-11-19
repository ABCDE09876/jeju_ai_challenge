# 24문항 벤치마크 결과

- 엔진: 로컬 신경-기호 엔진(결정론, force_offline=True, 캐시 미사용)
- 피크 파이썬 메모리 할당: 9.9 MB, 프로세스 RSS: 209.7 MB
- 정답 라벨 oracle 사전 검증: {'D1': True, 'D7': True, 'T1': True, 'T2': True, 'T3': True, 'T15': True}

| 지표 | Baseline 1 단순 평균 | Baseline 2 단일 표 조회 | Ours |
|---|---|---|---|
| 평균 착시 오판 건수(낮을수록 좋음) | 5 / 5 | 5 / 5 | 0 / 5 |
| 은폐 슬라이스 검출 건수 | 0 / 6 | 0 / 6 | 5 / 6 |
| 집계 변경·미측정 인과·분모 함정 차단 건수 | 0 / 4 | 0 / 4 | 4 / 4 |
| 정상 주장·대조군 오경보 건수(낮을수록 좋음) | 1 / 5 | 1 / 5 | 0 / 5 |
| 판정 보류(Abstention) 정확 건수 | 0 / 4 | 0 / 4 | 4 / 4 |
| Dev 정답(8) | 1 / 8 | 1 / 8 | 8 / 8 |
| Held-out Test 정답(16) | 3 / 16 | 3 / 16 | 15 / 16 |
| 전체 정답(24) | 4 / 24 | 4 / 24 | 23 / 24 |
| 평균 응답 시간(ms) | 7.1 | 7.5 | 284.0 |

## 문항별 결과(실패·보류 포함 전체 공개)

| ID | 구분 | 범주 | Ours 판정 | 통과 | 근거 | B1 | B2 |
|---|---|---|---|---|---|---|---|
| D1 | dev | A | refuted | O | 판정 refuted | confirmed(X) | confirmed(X) |
| D2 | dev | A | no_reversal | O | 귀가 경계 20시 (정답 [20, 21]) | no_reversal(X) | no_reversal(X) |
| D3 | dev | B | masked_overload | O | 제주민속오일장 정류소 승하차 · 장날(2·7일) · 09~13시 (2.24배) | none(X) | none(X) |
| D4 | dev | B | masked_overload | O | 삼화지구 정류소 승하차 · 비 오는 평일 · 17~19시 (1.78배) | none(X) | none(X) |
| D5 | dev | C | unmeasured_cause | O | 단절 월 ['201911'] (정답 201911) | confirmed(X) | confirmed(X) |
| D6 | dev | C | structural_break | O | 단절 월 ['202010'] (정답 202010) | confirmed(X) | confirmed(X) |
| D7 | dev | C | confirmed | O | 판정 confirmed | confirmed(O) | confirmed(O) |
| D8 | dev | C | abstain | O | 판정 보류 + 현장 확인 질문 | no_reversal(X) | no_reversal(X) |
| T1 | test | A | refuted | O | 판정 refuted | confirmed(X) | confirmed(X) |
| T2 | test | A | reversal | O | 판정 reversal | no_reversal(X) | no_reversal(X) |
| T3 | test | A | no_reversal | O | 판정 no_reversal | no_reversal(O) | no_reversal(O) |
| T4 | test | A | no_reversal | O | 귀가 경계 20시 (정답 [20, 21]) | no_reversal(X) | no_reversal(X) |
| T5 | test | A | abstain | O | 판정 보류 + 현장 확인 질문 | no_reversal(X) | no_reversal(X) |
| T6 | test | A | abstain | O | 판정 보류 + 현장 확인 질문 | no_reversal(X) | no_reversal(X) |
| T7 | test | B | masked_overload | O | 함덕해수욕장 정류소 승하차 · 7~8월 주말 · 하루 종일(06~23시) (2.09배) | none(X) | none(X) |
| T8 | test | B | masked_overload | O | 노형오거리 정류소 승하차 · 금요일 · 21~23시 (1.80배) | none(X) | none(X) |
| T9 | test | B | masked_overload | O | 제주국제공항 정류소 승하차 · 일요일 · 17~20시 (1.54배) | none(X) | none(X) |
| T10 | test | B | inconclusive | X | 판정 inconclusive (정답 ['masked_overload']) | none(X) | none(X) |
| T11 | test | B | inconclusive | O | 판정 inconclusive | confirmed(X) | confirmed(X) |
| T12 | test | B | inconclusive | O | 판정 inconclusive | refuted(O) | refuted(O) |
| T13 | test | C | thin_denominator | O | 판정 thin_denominator | confirmed(X) | confirmed(X) |
| T14 | test | C | unmeasured_cause | O | 판정 unmeasured_cause | no_reversal(X) | no_reversal(X) |
| T15 | test | C | confirmed | O | 판정 confirmed | confirmed(O) | confirmed(O) |
| T16 | test | C | abstain | O | 판정 보류 + 현장 확인 질문 | refuted(X) | refuted(X) |
