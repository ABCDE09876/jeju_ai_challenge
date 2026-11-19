# 실제 Claude 경로 40문항 시험 결과 (1차, 수정 전 코드)

> 1차 실행(공식, 수정 전 코드). 이 실행에서 찾은 결함(같은 도구를 여러 번 부르면 마지막 결과만 남음)을 고친 뒤 같은 40문항을 다시 돌린 2차 실행은 live_report.json 입니다.

- 실행: 2026-11-19 00:22:58 · 모델 claude-opus-5-5 · 문항마다 새 연결(5턴 후속 질문은 한 대화), 캐시 미사용
- 기준표 통과: 26 / 40 (실행 오류 0건)
- Claude 모드로 답함 39, 결론·세 구획·현장 확인 질문 40, 해설 숫자 모두 도구 근거 28, 판정 일치 38
- 벤치마크 24문항 정답: Claude 경로 22 / 24 · 같은 실행의 로컬 엔진 23 / 24
- 판정이 로컬 엔진과 같음: 40 / 40 · 해설 숫자 도구 근거 일치 953 / 976
- 답변 시간: 중앙값 36.0초, p95 58.9초, 범위 3.6~65.3초
- 토큰: 입력 874,268 · 출력 146,973 (요청 130회)
- 폴백: ['자동 폴백: 응답 지연(3.5초 내 첫 응답 없음)']

| 문항 | 묶음 | 질문 | 모드 | 판정 | 로컬 판정 | 정답·대조 | 숫자 | 초 | 통과 |
|---|---|---|---|---|---|---|---|---|---|
| B-D1 | benchmark | 애월이 아라동보다 살기 활기찬 동네라던데, 도민 기준으 | claude | refuted | refuted | 정답 | 31/31 | 37.1 | 예 |
| B-D2 | benchmark | 삼화지구로 이사 가면 퇴근 버스 괜찮을까? 아라동이랑  | claude | no_reversal | no_reversal | 오답 | 35/35 | 50.8 | 아니오 |
| B-D3 | benchmark | 도두동 오일장 정류소는 평균 이용객이 적어서 버스 증편 | claude | masked_overload | masked_overload | 정답 | 19/19 | 37.3 | 예 |
| B-D4 | benchmark | 비 오는 평일 퇴근길에 삼화지구 정류소가 언제 붐벼? | claude | masked_overload | masked_overload | 정답 | 31/31 | 34.8 | 예 |
| B-D5 | benchmark | 한림읍 음식물쓰레기가 2019년 12월에 10월보다 5 | claude | unmeasured_cause | unmeasured_cause | 정답 | 13/13 | 42.7 | 예 |
| B-D6 | benchmark | 제주 공공와이파이 이용자가 2020년 10월에 반토막  | claude | structural_break | structural_break | 정답 | 4/4 | 33.1 | 예 |
| B-D7 | benchmark | 노형동은 저녁에 도민 유동인구가 연동보다 많다는데 맞아 | claude | confirmed | confirmed | 정답 | 30/30 | 33.2 | 예 |
| B-D8 | benchmark | 노형동 집값 비싸? | claude | abstain | abstain | 정답 | 17/17 | 26.2 | 예 |
| B-T1 | benchmark | 중문동이 대륜동보다 사람이 훨씬 많아서 살기 활기찬 동 | claude | refuted | refuted | 정답 | 34/34 | 33.6 | 예 |
| B-T2 | benchmark | 일도1동이랑 삼도1동 중에 밤 9시 이후에 사람 사는  | claude | reversal | reversal | 정답 | 23/24 | 26.8 | 아니오 |
| B-T3 | benchmark | 노형동이랑 외도동 중 퇴근 후 생활하기 어디가 나아? | claude | no_reversal | no_reversal | 정답 | 33/33 | 41.5 | 예 |
| B-T4 | benchmark | 외도동이랑 삼화지구 중에 퇴근하고 버스로 귀가하기 어디 | claude | no_reversal | no_reversal | 정답 | 23/25 | 32.2 | 아니오 |
| B-T5 | benchmark | 작년 겨울 아라동이랑 노형동 중 밤에 어디가 더 붐볐어 | claude | abstain | abstain | 정답 | 26/26 | 37.8 | 예 |
| B-T6 | benchmark | 애월읍 아파트 월세 시세는 어느 정도야? | claude | abstain | abstain | 정답 | 7/7 | 20.0 | 예 |
| B-T7 | benchmark | 함덕 버스 정류소가 평소엔 한산한데 숨은 혼잡 시간대가 | claude | masked_overload | masked_overload | 정답 | 18/19 | 29.9 | 아니오 |
| B-T8 | benchmark | 노형오거리 정류소는 평균 이용객이 많지 않아 심야 증편 | claude | masked_overload | masked_overload | 정답 | 27/27 | 56.4 | 예 |
| B-T9 | benchmark | 제주공항 버스 정류소는 언제 제일 붐벼? 평균 말고 숨 | claude | masked_overload | masked_overload | 정답 | 17/17 | 36.5 | 예 |
| B-T10 | benchmark | 동문시장 앞 정류소는 이용객이 평균적이라 혼잡 민원이  | claude | inconclusive | inconclusive | 오답 | 17/21 | 65.3 | 아니오 |
| B-T11 | benchmark | 외도동 외도부영 정류소는 화요일 오전에 평소보다 붐빈다 | claude | inconclusive | inconclusive | 정답 | 19/21 | 42.4 | 아니오 |
| B-T12 | benchmark | 첨단과학기술단지 정류소는 오일장 장날에 평소보다 붐빈다 | claude | inconclusive | inconclusive | 정답 | 50/51 | 54.6 | 아니오 |
| B-T13 | benchmark | 전기차 충전 대기시간이 평균 20분이라는 보도, 믿어도 | claude | thin_denominator | thin_denominator | 정답 | 15/15 | 38.1 | 예 |
| B-T14 | benchmark | 비가 많이 와서 연동 카드 매출이 줄었다는데 맞아? | claude | unmeasured_cause | unmeasured_cause | 정답 | 7/7 | 27.5 | 예 |
| B-T15 | benchmark | 아라동은 저녁 시간에 도민이 삼양동보다 많다던데 맞아? | claude | confirmed | confirmed | 정답 | 33/33 | 46.4 | 예 |
| B-T16 | benchmark | 2024년 삼화지구 버스 이용객이 2025년보다 늘었다 | claude | abstain | abstain | 정답 | 12/12 | 29.3 | 예 |
| S01 | scenario | 동문시장 쪽 일도1동이랑 이도1동 중에 밤 9시 이후  | claude | reversal | reversal | 같음 | 21/21 | 30.0 | 예 |
| S02 | scenario | 오일장 장날 몇 시에 가면 덜 붐벼? 어르신 모시고 가 | claude | boundary | boundary | 같음 | 18/19 | 33.3 | 아니오 |
| S03 | scenario | 신제주랑 구제주 중 퇴근 후 저녁 생활 어디가 나아? | claude | no_reversal | no_reversal | 같음 | 28/28 | 46.3 | 예 |
| S04 | scenario | 도두동 오일장 정류소는 평균 이용객이 적어서 버스 증편 | claude | masked_overload | masked_overload | 같음 | 16/17 | 38.2 | 아니오 |
| S05 | scenario | 버스 이용이 줄어든 건 전기차 보급 때문이라는 기사 맞 | claude | unmeasured_cause | unmeasured_cause | 같음 | 19/19 | 59.0 | 예 |
| S06 | scenario | 전기차 충전 대기시간은 평균 17분이라 짧다는 기사 맞 | claude | thin_denominator | thin_denominator | 같음 | 24/24 | 44.4 | 예 |
| P01 | previous_live | 노형똥 퇴근 버스 어때? | local | no_reversal | no_reversal | 같음 | 19/19 | 3.6 | 아니오 |
| P02 | previous_live | 우도면 새벽 3시 비 오는 장날 어르신 버스 붐벼? | claude | no_reversal | no_reversal | 같음 | 11/11 | 36.8 | 예 |
| P03 | previous_live | '; DROP TABLE regions; -- | claude | insufficient_area | insufficient_area | 같음 | 7/7 | 9.3 | 예 |
| P04 | previous_live | 작년 겨울 노형동 밤 버스 붐볐어? | claude | abstain | abstain | 같음 | 26/26 | 35.4 | 예 |
| M01 | multi_turn | 삼화지구 퇴근 버스 어때? | claude | no_reversal | no_reversal | 같음 | 44/44 | 29.4 | 예 |
| M02 | multi_turn | 그럼 8시 이후는? | claude | no_boundary | no_boundary | 같음 | 34/37 | 29.4 | 아니오 |
| M03 | multi_turn | 비 오는 날에는? | claude | no_boundary | no_boundary | 같음 | 39/41 | 25.3 | 아니오 |
| M04 | multi_turn | 거기 말고 노형동이랑 비교해 줘 | claude | no_boundary | no_boundary | 같음 | 41/42 | 45.8 | 아니오 |
| M05 | multi_turn | 주말엔? | claude | no_boundary | no_boundary | 같음 | 37/41 | 28.7 | 아니오 |
| E01 | mixed_language | 노형동 vs 연동 which is better at n | claude | no_reversal | no_reversal | 같음 | 28/28 | 45.3 | 예 |
