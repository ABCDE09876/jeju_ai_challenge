# 실제 Claude 경로 40문항 시험 결과

- 실행: 2026-11-19 01:02:29 · 모델 claude-opus-5-5 · 문항마다 새 연결(5턴 후속 질문은 한 대화), 캐시 미사용
- 기준표 통과: 34 / 40 (실행 오류 0건)
- Claude 모드로 답함 38, 결론·세 구획·현장 확인 질문 40, 해설 숫자 모두 도구 근거 37, 판정 일치 39
- 벤치마크 24문항 정답: Claude 경로 23 / 24 · 같은 실행의 로컬 엔진 23 / 24
- 판정이 로컬 엔진과 같음: 40 / 40 · 해설 숫자 도구 근거 일치 878 / 881
- 답변 시간: 중앙값 34.6초, p95 57.6초, 범위 3.6~70.5초
- 토큰: 입력 890,364 · 출력 142,388 (요청 130회)
- 폴백: ['자동 폴백: 응답 지연(3.5초 내 첫 응답 없음)', '자동 폴백: 응답 지연(3.5초 내 첫 응답 없음)']

| 문항 | 묶음 | 질문 | 모드 | 판정 | 로컬 판정 | 정답·대조 | 숫자 | 초 | 통과 |
|---|---|---|---|---|---|---|---|---|---|
| B-D1 | benchmark | 애월이 아라동보다 살기 활기찬 동네라던데, 도민 기준으 | claude | refuted | refuted | 정답 | 32/32 | 38.7 | 예 |
| B-D2 | benchmark | 삼화지구로 이사 가면 퇴근 버스 괜찮을까? 아라동이랑  | claude | no_reversal | no_reversal | 정답 | 30/30 | 49.4 | 예 |
| B-D3 | benchmark | 도두동 오일장 정류소는 평균 이용객이 적어서 버스 증편 | claude | masked_overload | masked_overload | 정답 | 16/16 | 30.5 | 예 |
| B-D4 | benchmark | 비 오는 평일 퇴근길에 삼화지구 정류소가 언제 붐벼? | claude | masked_overload | masked_overload | 정답 | 33/33 | 47.2 | 예 |
| B-D5 | benchmark | 한림읍 음식물쓰레기가 2019년 12월에 10월보다 5 | claude | unmeasured_cause | unmeasured_cause | 정답 | 12/12 | 38.8 | 예 |
| B-D6 | benchmark | 제주 공공와이파이 이용자가 2020년 10월에 반토막  | claude | structural_break | structural_break | 정답 | 7/7 | 33.1 | 예 |
| B-D7 | benchmark | 노형동은 저녁에 도민 유동인구가 연동보다 많다는데 맞아 | claude | confirmed | confirmed | 정답 | 26/26 | 39.0 | 예 |
| B-D8 | benchmark | 노형동 집값 비싸? | claude | abstain | abstain | 정답 | 16/16 | 24.3 | 예 |
| B-T1 | benchmark | 중문동이 대륜동보다 사람이 훨씬 많아서 살기 활기찬 동 | claude | refuted | refuted | 정답 | 32/32 | 37.8 | 예 |
| B-T2 | benchmark | 일도1동이랑 삼도1동 중에 밤 9시 이후에 사람 사는  | claude | reversal | reversal | 정답 | 20/20 | 35.5 | 예 |
| B-T3 | benchmark | 노형동이랑 외도동 중 퇴근 후 생활하기 어디가 나아? | claude | no_reversal | no_reversal | 정답 | 31/31 | 43.9 | 예 |
| B-T4 | benchmark | 외도동이랑 삼화지구 중에 퇴근하고 버스로 귀가하기 어디 | claude | no_reversal | no_reversal | 정답 | 20/20 | 31.1 | 예 |
| B-T5 | benchmark | 작년 겨울 아라동이랑 노형동 중 밤에 어디가 더 붐볐어 | claude | abstain | abstain | 정답 | 21/21 | 28.3 | 예 |
| B-T6 | benchmark | 애월읍 아파트 월세 시세는 어느 정도야? | claude | abstain | abstain | 정답 | 11/11 | 19.9 | 예 |
| B-T7 | benchmark | 함덕 버스 정류소가 평소엔 한산한데 숨은 혼잡 시간대가 | claude | masked_overload | masked_overload | 정답 | 17/18 | 35.0 | 아니오 |
| B-T8 | benchmark | 노형오거리 정류소는 평균 이용객이 많지 않아 심야 증편 | claude | masked_overload | masked_overload | 정답 | 24/24 | 55.3 | 예 |
| B-T9 | benchmark | 제주공항 버스 정류소는 언제 제일 붐벼? 평균 말고 숨 | claude | masked_overload | masked_overload | 정답 | 17/17 | 33.8 | 예 |
| B-T10 | benchmark | 동문시장 앞 정류소는 이용객이 평균적이라 혼잡 민원이  | claude | inconclusive | inconclusive | 오답 | 31/31 | 70.5 | 아니오 |
| B-T11 | benchmark | 외도동 외도부영 정류소는 화요일 오전에 평소보다 붐빈다 | claude | inconclusive | inconclusive | 정답 | 23/23 | 51.6 | 예 |
| B-T12 | benchmark | 첨단과학기술단지 정류소는 오일장 장날에 평소보다 붐빈다 | claude | inconclusive | inconclusive | 정답 | 29/29 | 57.7 | 예 |
| B-T13 | benchmark | 전기차 충전 대기시간이 평균 20분이라는 보도, 믿어도 | local | thin_denominator | thin_denominator | 정답 | 14/14 | 3.6 | 아니오 |
| B-T14 | benchmark | 비가 많이 와서 연동 카드 매출이 줄었다는데 맞아? | claude | unmeasured_cause | unmeasured_cause | 정답 | 12/12 | 44.1 | 예 |
| B-T15 | benchmark | 아라동은 저녁 시간에 도민이 삼양동보다 많다던데 맞아? | claude | confirmed | confirmed | 정답 | 31/31 | 39.9 | 예 |
| B-T16 | benchmark | 2024년 삼화지구 버스 이용객이 2025년보다 늘었다 | claude | abstain | abstain | 정답 | 13/13 | 23.9 | 예 |
| S01 | scenario | 동문시장 쪽 일도1동이랑 이도1동 중에 밤 9시 이후  | local | reversal | reversal | 같음 | 15/15 | 10.4 | 아니오 |
| S02 | scenario | 오일장 장날 몇 시에 가면 덜 붐벼? 어르신 모시고 가 | claude | boundary | boundary | 같음 | 22/23 | 36.8 | 아니오 |
| S03 | scenario | 신제주랑 구제주 중 퇴근 후 저녁 생활 어디가 나아? | claude | no_reversal | no_reversal | 같음 | 28/28 | 32.1 | 예 |
| S04 | scenario | 도두동 오일장 정류소는 평균 이용객이 적어서 버스 증편 | claude | masked_overload | masked_overload | 같음 | 18/18 | 34.7 | 예 |
| S05 | scenario | 버스 이용이 줄어든 건 전기차 보급 때문이라는 기사 맞 | claude | unmeasured_cause | unmeasured_cause | 같음 | 27/27 | 48.8 | 예 |
| S06 | scenario | 전기차 충전 대기시간은 평균 17분이라 짧다는 기사 맞 | claude | thin_denominator | thin_denominator | 같음 | 20/21 | 41.1 | 아니오 |
| P01 | previous_live | 노형똥 퇴근 버스 어때? | claude | no_reversal | no_reversal | 같음 | 21/21 | 27.7 | 예 |
| P02 | previous_live | 우도면 새벽 3시 비 오는 장날 어르신 버스 붐벼? | claude | no_reversal | no_reversal | 같음 | 11/11 | 32.6 | 예 |
| P03 | previous_live | '; DROP TABLE regions; -- | claude | insufficient_area | insufficient_area | 같음 | 7/7 | 10.4 | 예 |
| P04 | previous_live | 작년 겨울 노형동 밤 버스 붐볐어? | claude | abstain | abstain | 같음 | 27/27 | 34.5 | 예 |
| M01 | multi_turn | 삼화지구 퇴근 버스 어때? | claude | no_reversal | no_reversal | 같음 | 31/31 | 39.5 | 예 |
| M02 | multi_turn | 그럼 8시 이후는? | claude | no_boundary | no_boundary | 같음 | 25/25 | 33.4 | 예 |
| M03 | multi_turn | 비 오는 날에는? | claude | no_boundary | no_boundary | 같음 | 30/30 | 31.0 | 예 |
| M04 | multi_turn | 거기 말고 노형동이랑 비교해 줘 | claude | no_boundary | no_boundary | 같음 | 29/29 | 32.1 | 예 |
| M05 | multi_turn | 주말엔? | claude | no_boundary | no_boundary | 같음 | 24/24 | 28.6 | 예 |
| E01 | mixed_language | 노형동 vs 연동 which is better at n | claude | no_reversal | no_reversal | 같음 | 25/25 | 29.5 | 예 |
