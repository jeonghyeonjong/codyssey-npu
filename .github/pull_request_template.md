## 변경 요약 (Summary)
- 추가/수정된 도형 라벨 또는 기능 요약: (예: `Circle` 표준 라벨 및 `'o'`, `'circle'` 별칭 추가)
- 관련 클래스 및 파일: `main.py` (`LabelNormalizer`, `DatasetRepository`, `SimulatorApp`), `tests/test_npu_simulator.py`

## 신규 라벨·판정 규칙 확장 체크리스트 (Extension & Review Checklist)
- [ ] `LabelNormalizer`에 표준 라벨 상수 선언 및 `_LABEL_MAP` 소문자 별칭 등록 완료
- [ ] `LabelNormalizer.ALLOWED_ALIAS_ONELINE` 한 줄 별칭 안내 문구 동기화 완료
- [ ] `DatasetRepository.load_normalized_filters`의 필수/선택 라벨 검증 반영 완료
- [ ] `SimulatorApp` 판정 출력 시 `format_judgment_line(..., epsilon)` 일원화 포맷 유지 확인
- [ ] `epsilon` 허용오차(`1e-9`) 경계 동점 케이스(`status_code="tie"`, `severity="LOW"`) 회귀 영향 점검 완료

## 자동화 테스트 및 리포트 검증 결과 (Automated Test & Report Verification)
- [ ] `python -m unittest tests/test_npu_simulator.py -v` 전체 테스트 통과 (`OK`)
- [ ] 신규 라벨 단위 테스트(`test_custom_new_label_extension_template` 복제본) 추가 및 정규화·2D/1D MAC 일치성 확인
- [ ] `python main.py` 모드 2 실행 시 `report_state.json` 동기화 생성 및 최소 줄 수(`>= 10줄`) 검증 통과
