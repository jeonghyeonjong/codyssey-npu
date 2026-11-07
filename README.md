# Mini NPU Simulator — 순수 파이썬 MAC 연산 및 패턴 판별 시뮬레이터

## 1. 프로젝트 개요 및 핵심 배경 원리

외부 수치 연산 프레임워크(`NumPy`, `pandas` 등) 없이 **순수 파이썬 표준 라이브러리와 반복문만으로 인공신경망·NPU의 핵심 연산인 MAC(Multiply-Accumulate)을 구현하고 성능을 정량 분석하는 콘솔 시뮬레이터**입니다.
`3×3` 대화형 필터·패턴 실험(모드 1), `data.json` 다중 크기(`5×5`·`13×13`·`25×25`) 일괄 검증 및 리포트 동기화 저장(모드 2), 동적 패턴 생성기·1차원 메모리 최적화·멀티스레드 병목 실험(모드 3)을 통합 제공합니다.

- **원격 저장소 URL**: https://github.com/jeonghyeonjong/codyssey-npu.git
- **저장소 공개 여부**: Public Repository
- **개발 및 실행 환경**:
  - OS / Shell: Windows 11 / Git Bash
  - Language: Python 3.15.2 (Python 3.8 이상 공식 요구조건 충족, 순수 표준 라이브러리 `json`, `time`, `math`, `os`, `shutil`, `sys`, `concurrent.futures`만 사용)
  - Version Control: Git / GitHub (`feature/mac-engine` 기능 브랜치 `--no-ff` 병합 및 `clone` / `pull` 원격 동기화 완비)

### 시각 패턴의 배열 표현, MAC 유사도 전제 조건 및 NPU의 필요성
1. **2차원 숫자 배열 표현**: 컴퓨터는 십자가(`+`)나 대각선(`X`) 형태를 시각적으로 인식하지 못하므로 픽셀 활성화 여부를 `0.0`과 `1.0`(또는 실수 가중치)로 나타낸 $N \times N$ 2차원 배열로 변환해 처리합니다.
2. **MAC 연산과 필터 가중치 총합 정규화 전제**: 입력 패턴 $P$와 기준 필터 $F$의 같은 좌표 원소끼리 곱해 누적 합산하는 $\text{Score} = \sum_{r=0}^{N-1}\sum_{c=0}^{N-1} P_{r,c} \cdot F_{r,c}$ 연산을 MAC이라 하며, 활성 위치가 일치할수록 점수가 커지므로 형태적 유사도(내적 상관도) 지표로 작동합니다.
   - **원시 내적 전제와 정규화 권장 기준**: 본 시뮬레이터의 기본 판정 모드는 과제 명세에 따라 `data.json`의 필터 가중치를 변형하지 않는 **비정규화 원시 MAC 점수**를 기준으로 판정합니다. 다만 원시 내적은 각 필터의 가중치 총합($\sum |F_{r,c}|$) 스케일에 비례하므로, 실무 패턴 분류기 구축이나 동점 케이스 재검증 시에는 두 필터의 절댓값 총합을 `1.0`으로 맞추는 **L1 가중치 합 정규화($\hat{F} = F / \sum |F|$)**를 선행하는 것을 권장하며, 코드 내에 보조 재검증 메서드(`MacEngine.normalize_filter_l1`)를 함께 구현해 두었습니다.
3. **CPU 직렬 처리의 한계와 NPU의 탄생 배경**: 실제 AI 모델은 수백×수백 필터 수천 개를 적용해 이미지 한 장당 수억 회의 단순 반복 MAC 연산을 수행합니다. 복잡한 분기·제어 명령의 순차 실행에 최적화된 CPU 직렬 파이프라인(특히 파이썬 환경에서는 GIL 제약까지 중첩됨)으로는 대규모 MAC 처리량에 병목이 발생하므로, 수천 개의 MAC 연산 셀을 하드웨어 격자(Systolic Array)로 배치해 동시에 병렬 계산하는 **NPU(Neural Processing Unit)**가 필수적입니다.

---

## 2. 디렉터리 구조, 실행 방법 및 운영 모드 가이드

### 1) 디렉터리 구조와 파일별 역할

    codyssey-npu/
    ├── .gitignore                    # 캐시(__pycache__/), 임시 직렬화(*.tmp), 백업·손상본(*.bak, *.corrupt) 제외
    ├── main.py                       # 메인 실행 파일 (Matrix2D, LabelNormalizer, MacEngine 등 6개 독립 클래스)
    ├── data.json                     # 공식 평가용 다중 크기(size_5, size_13, size_25) 필터 및 패턴 데이터셋
    ├── report_state.json             # 모드 2 실행 완료 시 콘솔 요약과 동기화 생성되는 JSON 결과 리포트 (런타임 생성)
    ├── tests/
    │   ├── __init__.py               # 테스트 패키지 초기화
    │   └── test_npu_simulator.py     # MAC 정합성, 허용오차, 키·라벨 오류, 병목 실험 등 자동화 단위 테스트 9종
    ├── docs/                         # 실제 실행 화면 및 Git 워크플로우 증빙 스크린샷
    │   ├── mode1_interactive.png     # 모드 1: 원인별 입력 오류 안내, MAC 판정 및 3×3 성능 분석
    │   ├── mode2_json_analysis.png   # 모드 2: data.json 일괄 판정, 동점 L1 재검증 및 리포트 저장 화면
    │   ├── error_isolation.png       # 미로드 후보 표시·크기 불일치·비표준 라벨·비정상 키 격리 및 Ctrl+C 종료
    │   ├── bonus_1d_generator.png    # 보너스: N×N 패턴 생성기, 1D 평탄화 비교 및 스레드 병목 실험
    │   ├── git_commit_count.png      # 총 커밋 수 및 feature/mac-engine 브랜치 --no-ff 병합 그래프
    │   └── git_clone_pull.png        # git clone 및 git pull 원격 저장소 동기화 실습 기록
    └── README.md                     # 실행 방법, 포맷 매핑, 실패 원인 분석 및 시간 복잡도 기술 리포트

### 2) 실행 명령어 및 매개변수 설정 안내

`data.json`은 프로젝트 루트(`codyssey-npu/data.json`)에 위치하며 `__file__` 기준 절대 경로로 자동 탐색됩니다. 구동 즉시 상단 배너에 현재 적용 중인 허용오차(`epsilon = 1e-09`)와 반복 측정 횟수(`repeats = 1000회`)가 표시됩니다.

    $ cd /c/codyssey-workspace/codyssey-npu
    $ python main.py

    # 전체 단위 테스트 9종 실행
    $ python tests/test_npu_simulator.py -v

코드 내에서 허용오차나 반복 횟수를 변경해 실험할 때는 `main.py`의 `MacEngine(epsilon=1e-9, repeats=1000)` 생성자 인자 또는 `DEFAULT_EPSILON` / `DEFAULT_REPEATS` 상수를 수정하면 전체 출력 헤더와 판정 로직, 저장 리포트(`report_state.json`)에 자동 반영됩니다.

### 3) 운영 시나리오별 권장 실행 모드 가이드

| 실행 모드 | 동점 시 출력 표기 | 권장 운영 시나리오 및 활용 사례 |
| :--- | :--- | :--- |
| **모드 1 (`1. 사용자 입력 3x3`)** | `판정 불가 (\|A-B\| < 1e-09)` | 신규 필터 가중치 초안을 설계한 직후 소형 `3×3` 커널 단위로 점수 민감도와 동점 경계를 즉석에서 대화형 디버깅할 때 권장 |
| **모드 2 (`2. data.json 분석`)** | `UNDECIDED` → `FAIL (동점 규칙)` | 다중 해상도(`5×5`~`25×25`) 표준 데이터셋에 대한 **자동화 회귀 검증(Regression Test)**, CI 품질 게이트 점검 및 감사 로그(`report_state.json`) 보관이 필요한 운영 환경에 권장 |
| **모드 3 (`3. 보너스 생성기·최적화`)** | `2D vs 1D` 수치 정합성 대조 | 실측 데이터가 아직 없는 임의 크기($N \times N$, `3~51`)에서 합성 패턴을 생성해 메모리 접근 최적화율과 CPU 병목을 사전 프로파일링할 때 권장 |

### 4) `main.py` 콘솔 출력 포맷과 본 문서 샘플 간의 명시적 매핑

본 문서 5장의 모든 터미널 로그와 `docs/*.png` 이미지는 `python main.py` 실제 표준 출력(stdout)을 가공 없이 그대로 캡처한 결과이며, 각 출력 블록을 생성하는 코드 위치는 아래와 같이 1:1 매핑됩니다.

| 실행 구간 및 출력 포맷 | `main.py` 담당 클래스·메서드 | 출력 포맷 규칙 요약 |
| :--- | :--- | :--- |
| 상단 설정 배너 및 메뉴 | `SimulatorApp.print_menu` | `[현재 설정] 허용오차(epsilon): 1e-09 \| 성능 측정 반복수(repeats): 1000회` |
| 모드 1 행렬 입력 오류 안내 | `SimulatorApp.read_matrix_from_console` | `입력 형식 오류 [구체적 사유]: 각 줄에 3개의 숫자를 공백으로 구분해 입력하세요.` |
| 모드 1 `[3] MAC 결과` | `SimulatorApp.run_user_input_mode` | `A 점수: ...` / `B 점수: ...` / `연산 시간(평균/1000회): ... ms` / `판정: A\|B\|판정 불가` |
| 모드 2 `[2] 패턴 분석` | `SimulatorApp.evaluate_dataset` | `Cross 점수: ...` / `X 점수: ...` / `판정: Cross\|X\|UNDECIDED \| expected: ... \| PASS\|FAIL` |
| 공통 성능 분석 표 | `SimulatorApp.print_benchmark_table` | 헤더에 `(반복 측정 repeats=1000회 평균)` 명시 + `크기 \| 반복수(회) \| 평균 시간(ms) \| 1D 최적화(ms) \| 연산 횟수(N²)` |
| 모드 2 `[4] 결과 요약` 및 파일 동기화 | `SimulatorApp.evaluate_dataset` | `총 테스트 / 통과 / 실패` + 실패 목록 + 동점 L1 재검증 + `[리포트 동기화 저장] report_state.json` |

---

## 3. 객체지향(OOP) 클래스 설계 및 관심사 분리 구조

| 클래스 | 핵심 속성 및 메서드 | 단일 책임(SRP) 및 세부 역할 |
| :--- | :--- | :--- |
| **`Matrix2D`** | `size`, `get_value(r, c)`, `set_value(r, c, v)`, `to_flat_list()` | $N \times N$ 정방 배열 저장, 특정 좌표 읽기·쓰기, 비정방·`bool`·`NaN` 차단 및 1D 평탄화 변환 |
| **`LabelNormalizer`** | `CROSS`, `X`, `UNDECIDED`, `normalize(raw_label)` | 외부 필터 키(`cross`, `x`)와 기대값(`+`, `x`)을 표준 라벨(`Cross`, `X`)로 정규화 및 허용 별칭 안내 |
| **`MacEngine`** | `compute_mac_2d`, `compute_mac_1d`, `normalize_filter_l1`, `judge_scores`, `benchmark_*` | 순수 반복문 2D/1D MAC 연산, L1 필터 정규화, `epsilon` 동점 판정, 순수 연산 및 스레드 병목 프로파일링 |
| **`PatternGenerator`** | `generate_cross(size)`, `generate_x(size)`, `generate_pair(size)` | 임의 크기 $N \times N$ 십자가(`Cross`) 및 대각선(`X`) 행렬 자동 생성 |
| **`DatasetRepository`** | `extract_size_from_key`, `load_dataset`, `load_normalized_filters`, `save_report_atomic` | 키 규칙 검증·크기 추출, `.corrupt` 격리 및 `.bak` 복구, `report_state.json` 원자적(`os.replace`) 저장 |
| **`SimulatorApp`** | `read_matrix_from_console`, `run_user_input_mode`, `evaluate_dataset`, `run` | 원인별 입력 검증, 후보 필터 안내 기반 케이스 단위 `FAIL` 격리, 요약 리포트 출력 및 안전 종료 |

`MacEngine` 내부에는 파일 입출력(`open`)이나 콘솔 출력(`print`) 코드가 전혀 없도록 격리하여, 성능 측정 시 오직 순수 산술 함수 호출 구간만 측정되도록 보장했습니다.

---

## 4. 핵심 기술 원리: 키·라벨 규칙, 누적 합산 방식 및 `epsilon` 비교 정책

### 1) `data.json` 키 파싱 규칙 및 비정상 키 에러 안내 체계

`DatasetRepository.extract_size_from_key()`는 필터 키(`size_{N}`)와 패턴 키(`size_{N}_{idx}`)에서 정수 `N`을 추출해 매칭하며, 형식 위반 시 원인별 메시지와 권장 키 예시를 함께 출력합니다. 또한 패턴이 요구하는 `size_N` 필터가 없을 때는 **현재 메모리에 로드된 필터 후보 목록**을 함께 표시해 즉시 원인을 파악할 수 있게 했습니다.

| 검증 상황 | 입력 예시 | 출력되는 구체적 에러 메시지 및 가이드 |
| :--- | :--- | :--- |
| 접두사(`size_`) 누락 | `"badkey_5_1"` | `키 접두사('size_') 누락: 'badkey_5_1' (권장 예시: 필터 키 'size_5', 패턴 키 'size_5_1', 'size_13_2').` |
| 크기 `N` 비정수 지정 | `"size_abc_1"` | `크기 N 위치에 정수가 아닌 문자열('abc')이 지정되었습니다: 'size_abc_1' (권장 예시: ...).` |
| `0` 이하 크기 지정 | `"size_0_1"` | `크기 N(0)은 1 이상의 양의 정수여야 합니다: 'size_0_1' (권장 예시: ...).` |
| 패턴 일련번호 누락 | `"size_5"` (패턴 영역) | `패턴 식별자 일련번호('_{idx}')가 누락되었습니다: 'size_5' (권장 예시: ...).` |
| 매칭 필터 미존재 | `"size_9_1"` | `매칭되는 'size_9' 필터가 없습니다 (현재 로드된 필터 후보: [size_5, size_13, size_25]).` |

### 2) 표준 라벨 정규화 규칙 및 허용 표기 목록

`data.json`의 `filters` 영역(`"cross"`, `"x"`)과 `patterns.expected` 영역(`"+"`, `"x"`) 간 표기 불일치를 해소하기 위해 진입 즉시 내부 표준 라벨 2종(**`Cross`**, **`X`**)으로 변환합니다.

| 내부 표준 라벨 | 허용되는 외부 입력 표기 (앞뒤 공백·대소문자 무시) | 주요 사용 위치 | 비허용 입력 예시 및 출력 안내 |
| :--- | :--- | :--- | :--- |
| **`Cross`** | `"+"`, `"cross"`, `"CROSS"`, `" Cross "` | `filters.*.cross`, `patterns.*.expected` | `"triangle"`, `"circle"` 입력 시 → `알 수 없는 비표준 라벨입니다: 'triangle' (허용 표기 — Cross 계열: '+', 'cross', 'CROSS' / X 계열: 'x', 'X')` |
| **`X`** | `"x"`, `"X"`, `" x "` | `filters.*.x`, `patterns.*.expected` | 빈 문자열 `""` 입력 시 → `라벨 값이 비어 있거나 문자열이 아닙니다. (허용 표기 — ...)` |

### 3) 순차 반복문 누적(`for` loop) vs 내장 `sum()` 합산 방식의 구현 차이

파이썬에서 MAC 연산 누적 합을 작성할 때 **순차 `for` 루프(`total += p * f`)**와 **내장 `sum()` 함수**는 최신 파이썬(CPython 3.12+)에서 수치적으로 다르게 동작합니다.
- **순차 `for` 루프 누적 (`compute_mac_2d` 채택 방식)**: 실제 NPU 하드웨어 MAC 레지스터가 클럭마다 곱셈 결과를 누산기(Accumulator)에 순차 가산하는 동작과 정확히 일치하며, 외부·C 레벨 벡터 합산 도움 없이 순수 반복문 제약을 엄격히 준수합니다. 이 방식에서는 `0.1`을 9회 더할 때 이진 반올림 오차가 누적되어 `0.8999999999999999`가 반환됩니다.
- **내장 `sum()` / `math.fsum()` 방식**: CPython 3.12 이상부터 실수 `sum()` 내부에 C 레벨 Neumaier 보정 합산 알고리즘이 적용되어 `sum([0.1] * 9)`가 오차 보정된 `0.9`를 반환합니다. 하지만 행렬 크기나 연산 순서가 달라지면(`25×25` 등) 여전히 부동소수점 오차가 발생할 수 있으므로, 합산 구현 방식에 의존하지 않고 판정 단계에서 `epsilon` 허용오차 정책을 두는 것이 근본적인 공학적 해법입니다.

### 4) 절대 허용오차(`epsilon = 1e-9`) 선택 근거 및 값 변경 시 영향 범위

1. **상대오차(`rel_tol`) 대신 절대오차(`abs(a - b) < 1e-9`)를 선택한 근거**:
   - 두 점수가 `0.0` 근처일 때 상대오차 수식($|a-b| \le \text{rel\_tol} \cdot \max(|a|, |b|)$)은 분모가 0에 수렴해 허용 폭이 사라지는 문제가 발생합니다. 반면 본 시뮬레이터가 다루는 행렬 크기(`3×3`~`25×25`, 최대 625회 누적)와 가중치 스케일(`0.1`~`7.5`)에서 발생 가능한 IEEE-754 누적 반올림 오차는 최대 $10^{-15} \sim 10^{-14}$ 수준인 반면, 유의미한 도형 간 점수 차이는 최소 `0.1`(`10⁻¹`) 이상입니다. 따라서 그 정중앙인 **`1e-9`(`10⁻⁹`) 절대오차**는 부동소수점 잡음은 100% 흡수하면서도 실제 점수 차이는 절대 뭉개지 않는 6자리 이상의 안전 마진을 제공합니다.
2. **`epsilon` 값 변경 시 판정 영향 범위**:
   - **과도하게 작게 설정할 경우 (예: `epsilon < 1e-16`)**: 이진 반올림 오차(`1.11e-16`, `2.66e-15`)를 동점으로 잡아내지 못해 `size_5_1`, `size_13_2`, `size_25_1`처럼 수학적으로 동점인 케이스가 오차 방향에 따라 반대 도형으로 오판정됩니다.
   - **권장 안정 구간 (`1e-12 <= epsilon <= 1e-6`)**: 부동소수점 누적 오차만 정확히 `UNDECIDED`(`판정 불가`)로 분류하고 정상 패턴 차이는 왜곡 없이 판별합니다.
   - **과도하게 크게 설정할 경우 (예: `epsilon >= 0.5`)**: 미세한 가중치 차이를 가진 서로 다른 패턴 점수까지 모두 동점(`UNDECIDED`)으로 뭉개져 판별력(Sensitivity)이 상실됩니다.

### 5) 벤치마크 성능 측정(`time.perf_counter()`) 포함 및 제외 구간 명세

순수 연산 복잡도($O(N^2)$)가 I/O 지연이나 객체 변환 비용에 오염되지 않도록 측정 경계를 엄격히 분리했습니다.
- **측정에서 제외된 사전 준비 구간 (Excluded)**: ① `data.json` 파일 열기 및 JSON 파싱, ② 콘솔 사용자 입력(`input`) 및 결과 표 출력(`print`), ③ `PatternGenerator` 테스트 행렬 생성 및 `Matrix2D` 정방 검증, ④ 1D 성능 측정을 위한 **`pattern.to_flat_list()` 및 `filter_mat.to_flat_list()` 사전 평탄화 리스트 생성 비용**.
- **측정에 포함된 순수 연산 구간 (Included)**: 오직 `repeats`(`1000`회) 반복 루프 안에서 실행되는 `compute_mac_2d(pattern, filter_mat)` 또는 `compute_mac_1d(flat_p, flat_f)` 함수 호출 및 산술 곱셈·누적 구간(`((t_end - t_start) / repeats) * 1000.0 ms`).

---

## 5. 모드별 실제 실행 로그 및 `docs/` 증빙 이미지

### 1) 모드 1: `3×3` 입력 원인별 오류 안내(열 개수·비숫자 구분), MAC 판정 및 성능 표

    $ python main.py

    === Mini NPU Simulator ===
    [현재 설정] 허용오차(epsilon): 1e-09 | 성능 측정 반복수(repeats): 1000회

    [모드 선택]
    1. 사용자 입력 (3x3)
    2. data.json 분석
    3. 패턴 생성기 및 2D/1D 메모리 최적화 비교 (보너스)
    4. 종료
    선택: 1

    #----------------------------------------
    # [1] 필터 입력
    #----------------------------------------

    필터 A (3줄 입력, 공백 구분)
      1행: 0 1
    입력 형식 오류 [열 개수 불일치: 입력 2개 != 기대 3개]: 각 줄에 3개의 숫자를 공백으로 구분해 입력하세요.
      1행: 0 a 0
    입력 형식 오류 [숫자 파싱 실패('a')]: 각 줄에 3개의 숫자를 공백으로 구분해 입력하세요.
      1행: 0 1 0
      2행: 1 1 1
      3행: 0 1 0

    필터 B (3줄 입력, 공백 구분)
      1행: 1 0 1
      2행: 0 1 0
      3행: 1 0 1

    [OK] 필터 저장 확인 완료: 필터 A (3×3), 필터 B (3×3)

    #----------------------------------------
    # [2] 패턴 입력
    #----------------------------------------

    패턴 (3줄 입력, 공백 구분)
      1행: 1 0 1
      2행: 0 1 0
      3행: 1 0 1

    #----------------------------------------
    # [3] MAC 결과
    #----------------------------------------
    A 점수: 1
    B 점수: 5
    연산 시간(평균/1000회): 0.0025 ms
    판정: B

    #----------------------------------------
    # [4] 성능 분석 (반복 측정 repeats=1000회 평균)
    #----------------------------------------
    크기      반복수(회)      평균 시간(ms)       1D 최적화(ms)      연산 횟수(N²)
    ------------------------------------------------------------------
    3×3     1000        0.0009          0.0007          9

![Mode 1 Interactive Screenshot](docs/mode1_interactive.png)

### 2) 모드 2: `data.json` 일괄 판정, 동점 L1 재검증 및 `report_state.json` 동기화 저장

    선택: 2

    #----------------------------------------
    # [1] 필터 로드
    #----------------------------------------
    [OK] size_5 필터 로드 완료 (Cross, X)
    [OK] size_13 필터 로드 완료 (Cross, X)
    [OK] size_25 필터 로드 완료 (Cross, X)

    #----------------------------------------
    # [2] 패턴 분석 (라벨 정규화 적용)
    #----------------------------------------
    --- size_5_1 ---
    Cross 점수: 0.9000000000000000
    X 점수: 0.8999999999999999
    판정: UNDECIDED | expected: X | FAIL (동점 규칙)
    --- size_5_2 ---
    Cross 점수: 8.9
    X 점수: 0.1
    판정: Cross | expected: Cross | PASS
    --- size_13_1 ---
    Cross 점수: 0.3
    X 점수: 14.7
    판정: X | expected: X | PASS
    --- size_13_2 ---
    Cross 점수: 7.4999999999999973
    X 점수: 7.5000000000000000
    판정: UNDECIDED | expected: Cross | FAIL (동점 규칙)
    --- size_25_1 ---
    Cross 점수: 4.9000000000000004
    X 점수: 4.8999999999999986
    판정: UNDECIDED | expected: X | FAIL (동점 규칙)
    --- size_25_2 ---
    Cross 점수: 52.9
    X 점수: 0.1
    판정: Cross | expected: Cross | PASS

    #----------------------------------------
    # [3] 성능 분석 (반복 측정 repeats=1000회 평균)
    #----------------------------------------
    크기      반복수(회)      평균 시간(ms)       1D 최적화(ms)      연산 횟수(N²)
    ------------------------------------------------------------------
    3×3     1000        0.0009          0.0007          9
    5×5     1000        0.0018          0.0014          25
    13×13   1000        0.0099          0.0080          169
    25×25   1000        0.0499          0.0367          625

    #----------------------------------------
    # [4] 결과 요약
    #----------------------------------------
    총 테스트: 6개
    통과: 3개
    실패: 3개

    실패 케이스:
    - size_5_1: 동점(UNDECIDED) 처리 규칙에 따라 FAIL (|Cross-X| = 1.11e-16 < 1e-09, expected: X)
    - size_13_2: 동점(UNDECIDED) 처리 규칙에 따라 FAIL (|Cross-X| = 2.66e-15 < 1e-09, expected: Cross)
    - size_25_1: 동점(UNDECIDED) 처리 규칙에 따라 FAIL (|Cross-X| = 1.78e-15 < 1e-09, expected: X)

    [동점(UNDECIDED) 케이스 보조 재검증 — 필터 L1 가중치 정규화 시뮬레이션]
      * size_5_1: L1 가중치 합 정규화 시 Cross=0.1011, X=1.0000 -> X (expected: X)
      * size_13_2: L1 가중치 합 정규화 시 Cross=1.0000, X=0.5102 -> Cross (expected: Cross)
      * size_25_1: L1 가중치 합 정규화 시 Cross=0.0926, X=1.0000 -> X (expected: X)

    [리포트 동기화 저장] report_state.json (생성 시각: 2026-11-07 19:52:02)

![Mode 2 JSON Analysis Screenshot](docs/mode2_json_analysis.png)

### 3) 스키마·후보 목록·크기·라벨·키 오류 케이스 단위 `FAIL` 격리 및 `Ctrl+C` 안전 종료

![Error Isolation and Graceful Shutdown Screenshot](docs/error_isolation.png)

### 4) 보너스 모드: `7×7` 패턴 자동 생성기, 1D 평탄화 비교 및 스레드 병목 실험

![Bonus Pattern Generator and 1D Optimization Screenshot](docs/bonus_1d_generator.png)

---

## 6. 결과 리포트 (실패 원인 분석 · 동점 재검증 · 시간 복잡도 및 병목 분석)

### 1) 구조화 리포트(`report_state.json`) 생성 시점 및 콘솔 출력 동기화 구조

모드 2(`evaluate_dataset`) 실행 시 모든 케이스 분석과 성능 측정이 끝난 직후, 화면의 `# [4] 결과 요약`을 출력하는 동일한 `summary_payload` 딕셔너리를 `DatasetRepository.save_report_atomic()`에 전달하여 루트 경로의 **`report_state.json`**(직전 본은 `report_state.json.bak` 백업)으로 즉시 원자적 저장합니다. 동일한 메모리 객체를 콘솔 출력과 JSON 직렬화가 공유하므로 실행 화면 수치와 저장된 리포트 파일은 항상 100% 일치합니다.

### 2) `data.json` 실패 케이스 3건 원인 분리 진단 및 L1 정규화 재검증 절차

공식 `data.json` 평가 결과는 **총 6건 중 통과 3건(`size_5_2`, `size_13_1`, `size_25_2`), 실패 3건(`size_5_1`, `size_13_2`, `size_25_1`)**이며, 세 가지 관점으로 분리한 원인 진단은 다음과 같습니다.

| 실패 케이스 | 기대 라벨 | 원시 `Cross` 점수 | 원시 `X` 점수 | `\|Cross - X\|` | L1 정규화 재검증 결과 (`Cross` vs `X`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`size_5_1`** | `X` | `0.9000000000000000` | `0.8999999999999999` | `1.11e-16` (`UNDECIDED`) | `0.1011` vs `1.0000` → **`X` (일치)** |
| **`size_13_2`** | `Cross` | `7.4999999999999973` | `7.5000000000000000` | `2.66e-15` (`UNDECIDED`) | `1.0000` vs `0.5102` → **`Cross` (일치)** |
| **`size_25_1`** | `X` | `4.9000000000000004` | `4.8999999999999986` | `1.78e-15` (`UNDECIDED`) | `0.0926` vs `1.0000` → **`X` (일치)** |

1. **데이터·가중치 스키마 원인**: `size_5.cross` 중앙의 `0.9`($= 9 \times 0.1$), `size_13.x` 중앙의 `7.5`($= 25 \times 0.3$), `size_25.cross` 중앙의 `4.9`($= 49 \times 0.1$)에 반대 필터 전체 합과 똑같은 가중치가 집중되어, 중심 교차점 단 1칸 공유만으로 수학적 총점이 완전히 같아지도록 설계되어 있습니다.
2. **수치 비교·부동소수점 원인**: 순차 누적 과정에서 발생한 이진 반올림 오차로 인해 세 건 모두 정답 도형의 원시 점수가 미세하게 작게 계산되었으나, `abs(a - b) < 1e-9` 정책이 오차에 의한 반대 도형 오판정을 막고 정확히 `UNDECIDED`(동점)로 방어했습니다.
3. **동점 결과에 대한 추가 재검증 절차**: 모드 2 요약 하단에서 자동 수행되는 보조 재검증(`MacEngine.normalize_filter_l1`)처럼 각 필터를 자신의 가중치 절댓값 총합(`8.9`, `0.9`, `7.5`, `14.7`, `52.9`, `4.9`)으로 나누어 스케일을 `1.0`으로 정규화하면, 위 표 우측 열과 같이 중심점 단일 가중치 왜곡이 제거되어 **3건 모두 기대 라벨(`X`, `Cross`, `X`)로 명확히 분리됨**을 확인할 수 있습니다.

### 3) 이론적 $O(N^2)$ 배율 대 실측 시간 비율 해석 및 대형 행렬 최적화·병목 실험

`5×5`(`25`회 연산)를 기준점(`1.00×`)으로 삼아 이론적 연산량 증가 배율($N^2 / 25$)과 실제 측정 시간 배율($T(N) / T(5)$)의 비율을 비교한 정량 분석 표입니다.

| 크기 ($N \times N$) | 연산 횟수 ($N^2$) | 이론 연산량 배율 ($N^2 / 25$) | 2D 실측 평균(`ms`) | 2D 실측 배율 ($T / T_5$) | **실측 / 이론 일치비** | 원소당 시간 (`ns / op`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`3×3`** | `9` | `0.36×` | `0.0009 ms` | `0.50×` | `1.39` (호출 고정비 구간) | `100.0 ns` |
| **`5×5`** | `25` | `1.00×` | `0.0018 ms` | `1.00×` | **`1.00` (기준점)** | `72.0 ns` |
| **`13×13`** | `169` | `6.76×` | `0.0099 ms` | `5.50×` | **`0.81` ($O(N^2)$ 수렴)** | `58.6 ns` |
| **`25×25`** | `625` | `25.00×` | `0.0499 ms` | `27.72×` | **`1.11` ($O(N^2)$ 일치)** | `79.8 ns` |

- **$O(N^2)$ 정량 검증**: 한 변의 길이 $N$이 `5`에서 `25`로 5배 증가할 때 이론 연산량($N^2$)은 **25.00배** 증가하고 실측 평균 시간은 **27.72배**(`실측/이론 비율 = 1.11`) 증가하며, 원소 1개당 소요 시간(`ns/op`)이 약 `58~80 ns` 대역으로 일정하게 유지되어 시간 복잡도가 정확히 $O(N^2)$을 따름을 실증합니다.
- **CPU 직렬 vs 파이썬 다중 스레드 병목 실측 실험 (모드 3 수록)**:
  - 동일 조건(`25×25` 필터 64개 배치, 총 `40,000`회 MAC 연산)에서 측정한 결과 **단일 스레드 2D 직렬 `2.091 ms`**, **단일 스레드 1D 평탄화 직렬 `2.037 ms`** 대비 **`ThreadPoolExecutor` 4스레드 실행은 `7.503 ms`로 오히려 약 3.6배 지연**되었습니다.
  - 이는 파이썬 인터프리터의 **GIL(Global Interpreter Lock)**로 인해 CPU 연산 중심(CPU-bound) 바이트코드가 실제로는 한 번에 하나의 스레드에서만 실행되면서 스레드 전환 및 락 경합 비용만 추가되기 때문입니다.
- **$1000 \times 1000$ 이상 대형 입력 처리 시 단계별 최적화 우선순위**:
  1. **1순위 (메모리 레이아웃 및 알고리즘 최적화)**: 중첩 리스트 포인터 역참조를 없애는 **1D 연속 배열 평탄화(`to_flat_list` / `array('d')`)** 적용 및 필터 내 `0.0` 원소가 많은 경우 비영(Non-zero) 좌표만 곱하는 희소(Sparse) MAC 전환.
  2. **2순위 (단일 코어 C 레벨 SIMD 벡터화)**: 파이썬 바이트코드 루프 오버헤드를 제거하기 위해 C 확장 기반 연속 메모리 내적(AVX2/AVX-512 SIMD 명령어)으로 이관.
  3. **3순위 (멀티프로세스 및 NPU/GPU 하드웨어 병렬화)**: GIL 제약을 우회하는 `ProcessPoolExecutor` 공유 메모리 배치 분할, 최종적으로는 수천 개 MAC 셀을 동시 구동하는 GPU Tensor Core / NPU 가속기로 오프로딩.

---

## 7. 데이터 복구 정책 및 Git 브랜치·원격 동기화 증빙

- **케이스 단위 `FAIL` 격리 및 파일 단위 `.bak` 복구**: 개별 패턴의 크기·라벨·키 오류는 무중단 케이스 단위 `FAIL`로 기록하며, `data.json` 파일 자체가 파손되었을 때는 깨진 원본을 `data.json.corrupt`로 대피시킨 뒤 `data.json.bak`으로 복원합니다.

![Git Commit Count and Graph Screenshot](docs/git_commit_count.png)

![Git Clone and Pull Sync Screenshot](docs/git_clone_pull.png)

    $ git rev-list --count HEAD
    16

    jeonghyeonjong@DESKTOP MINGW64 /c/codyssey-workspace/codyssey-npu (main)
    $ git log --oneline --graph
    * (HEAD -> main, origin/main) docs: enrich README and screenshots with format mapping, parameter guides, and O(N^2) ratios
    * fdef9a9 feat: enhance error diagnostics, repeats column, L1 tie recheck, and GIL benchmark
    * 931c21b docs: attach git commit graph and clone/pull remote sync verification
    * 690987e docs: write comprehensive architecture, failure diagnosis, and O(N^2) report in README
    * fe9eab5 docs: add real CLI execution screenshots for Mode 1, Mode 2, error isolation, and bonus
    * 7e30902 test: add automated unit tests for MAC parity, epsilon ties, and error isolation
    * 8c4dace feat: implement Mode 2 data.json batch evaluation, failure isolator, and bonus mode
    * 0f89d61 feat: implement Mode 1 interactive 3x3 filter and pattern console analyzer
    * c18f1ad feat: implement DatasetRepository with key parser, .bak recovery, and atomic save
    * 5064128 feat: add PatternGenerator and O(N^2) multi-scale benchmark profiler
    *   d4b894e merge: merge branch 'feature/mac-engine' into main
    |\  
    | * 692dbb4 feat: add epsilon tie-breaking policy and 1D flattened MAC optimization
    | * 8611ad3 feat: implement pure-python 2D nested-loop MAC operation
    |/  
    * 22fbd2b feat: add LabelNormalizer for standard Cross and X label mapping
    * 6c8e869 feat: implement Matrix2D class with NxN validation and cell accessors
    * 82a7307 chore: initialize repository with .gitignore and official data.json

---

## 8. 트러블슈팅 및 문제 해결 기록

### 사례 1: CPython 3.12+ 내장 `sum()`의 보정 합산과 순차 반복문 MAC 간 부동소수점 오차 차이 규명
- **문제 현상**: 단위 테스트에서 `sum([0.1] * 9)`로 부동소수점 누적 오차를 재현했을 때 예상했던 `0.8999999999999999` 대신 정확한 `0.9`가 반환되어 순차 `for` 루프 MAC 결과와 불일치하는 현상을 발견했습니다.
- **원인 가설**: Python 3.12 이상(`3.15.2`)의 내장 `sum()`은 C 레벨 Neumaier 보정 합산을 수행하는 반면, 직접 작성한 2중 `for` 루프는 순수 IEEE-754 순차 가산을 수행하기 때문이라고 가설을 세웠습니다.
- **검증 확인**: 동일 배열을 내장 `sum()`과 순차 `for` 루프로 비교 실행해 `for` 루프에서만 `1.11e-16`~`2.66e-15` 반올림 오차가 그대로 드러남을 확인했습니다.
- **해결 조치**: 단위 테스트가 실제 MAC 엔진과 동일한 순차 루프를 검증하도록 맞추고, `abs(score_a - score_b) < 1e-9` 정책으로 오차 방향과 무관하게 일관된 동점(`UNDECIDED`) 판정을 보장했습니다.

### 사례 2: 미로드 필터 참조 및 크기 불일치 시 무중단 격리와 후보 필터 안내 개선
- **문제 현상**: `data.json`에 매칭되지 않는 크기 키(`size_9_1`)나 비정방/크기 불일치 패턴이 섞여 있을 때 단순 에러만 띄우거나 종료되면 디버깅에 필요한 컨텍스트가 부족했습니다.
- **원인 가설**: 예외 처리 단위가 케이스별로 분리되지 않았고, 매칭 실패 시 현재 메모리에 정상 로드된 필터 목록 정보가 에러 메시지에 포함되지 않았기 때문입니다.
- **검증 확인**: `size_9_1`, `3×3 vs 5×5`, 비표준 라벨, 접두사 누락 키를 동시 주입한 테스트 시나리오로 진단 정보를 점검했습니다.
- **해결 조치**: `evaluate_dataset()` 루프 내부에서 케이스 단위로 예외를 격리하고, 필터 매칭 실패 시 `(현재 로드된 필터 후보: [size_5, size_13, size_25])`를 함께 출력하도록 개선했습니다.

### 사례 3: Windows 기본 콘솔(`cp949`) 인코딩 호환성 확보
- **문제 현상**: 비 UTF-8 서브프로세스 파이프 환경(`cp949`)에서 특수 유니코드 체크 기호 출력 시 `UnicodeEncodeError` 발생 가능성을 확인했습니다.
- **원인 가설**: Windows 기본 코드페이지(`cp949`)에 없는 보충 유니코드 문자를 인코딩 재설정 없이 출력했기 때문입니다.
- **검증 확인**: 서브프로세스 표준 출력 환경에서 문자셋 호환 범위를 테스트했습니다.
- **해결 조치**: 상태 표기를 `[OK]` / `[ERR]`로 표준화하고 `main()` 진입 시 `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` 안전장치를 적용해 모든 터미널에서 동일하게 동작하도록 해결했습니다.

---

## 9. 요구사항 변경 관리, 신규 라벨 확장 체크리스트 및 테스트 템플릿 (Maintenance Matrix)

### 1) 변경 시나리오별 최우선 수정 파일·클래스·메서드 매핑 표

| 요구사항 변경 시나리오 | 최우선 수정 파일 | 담당 클래스 | 핵심 수정 메서드 / 상수 | 세부 변경 작업 내용 |
| :--- | :--- | :--- | :--- | :--- |
| **허용오차(`epsilon`) 또는 반복수(`repeats`) 변경** | `main.py` | `MacEngine` | `DEFAULT_EPSILON`, `DEFAULT_REPEATS`, `judge_scores` | 상수 또는 생성자 인자 조정 (메뉴 배너·성능표·리포트 자동 연동) |
| **신규 도형 라벨 추가** (예: 원형 `Circle`) | `main.py` | `LabelNormalizer`, `DatasetRepository`, `SimulatorApp` | `_LABEL_MAP`, `load_normalized_filters`, `evaluate_dataset` | 아래 2절의 5단계 확장 체크리스트 및 다중 후보 `max` 판정 적용 |
| **필터 가중치 L1 정규화를 기본 판정에 적용** | `main.py` | `MacEngine`, `DatasetRepository` | `normalize_filter_l1`, `load_normalized_filters` | 필터 로드 시점에 `normalize_filter_l1()`을 거쳐 정규화 행렬 저장 |
| **편향(`+ b`) 또는 슬라이딩 윈도우 합성곱 추가** | `main.py` | `MacEngine` | `compute_mac_2d` (및 신규 `convolve_2d`) | 누적 합에 `bias` 가산 또는 부분 영역 순회 합성곱 메서드 확장 |

### 2) 신규 도형 라벨(예: `Circle`) 추가 시 5단계 수정 체크리스트 및 코드·테스트 템플릿

새로운 도형 라벨(`Circle`, 표기 `'o'` / `'circle'`)을 추가할 때는 아래 체크리스트 순서대로 수정하고 테스트 템플릿을 복제해 검증합니다.

- [ ] **Step 1**: `LabelNormalizer`에 `CIRCLE = "Circle"` 상수 추가 및 `_LABEL_MAP`에 `"o": CIRCLE, "circle": CIRCLE` 등록 (`allowed_aliases_summary` 문구 갱신)
- [ ] **Step 2**: `DatasetRepository.load_normalized_filters()`의 필수/선택 필터 키 검증 목록에 `LabelNormalizer.CIRCLE` 반영
- [ ] **Step 3**: `SimulatorApp.evaluate_dataset()`에서 로드된 전체 표준 라벨별 MAC 점수를 딕셔너리로 계산한 뒤 최고 점수 라벨과 2위 점수 차이가 `epsilon` 미만이면 `UNDECIDED`로 판정하도록 확장
- [ ] **Step 4**: (선택) `PatternGenerator.generate_circle(size)` 패턴 생성 메서드 추가
- [ ] **Step 5**: `tests/test_npu_simulator.py`에 아래 복제용 테스트 템플릿을 추가해 정규화·판정·동점 경계를 회귀 검증

    # [신규 라벨(Circle) 확장 코드 예시 — main.py > LabelNormalizer]
    CIRCLE = "Circle"
    _LABEL_MAP = {
        "+": CROSS,
        "cross": CROSS,
        "x": X,
        "o": CIRCLE,
        "circle": CIRCLE,
    }

    # [복제 가능한 단위 테스트 템플릿 — tests/test_npu_simulator.py 에 복사하여 사용]
    def test_custom_new_label_extension_template(self):
        # 1. 신규 라벨 별칭 정규화 검증
        self.assertEqual(LabelNormalizer.normalize("o"), "Circle")
        self.assertEqual(LabelNormalizer.normalize(" CIRCLE "), "Circle")
        # 2. 정방 행렬 생성 및 2D/1D MAC 일치성 검증
        sample_pattern = Matrix2D([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        sample_filter = Matrix2D([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        engine = MacEngine(epsilon=1e-9)
        self.assertAlmostEqual(
            engine.compute_mac_2d(sample_pattern, sample_filter),
            engine.compute_mac_1d(sample_pattern.to_flat_list(), sample_filter.to_flat_list()),
        )

---

## 10. 공식 요구사항 및 품질 기준 달성 요약

- [x] **순수 표준 라이브러리 구현**: 외부 벡터화 라이브러리 없이 순수 반복문 2D/1D MAC 연산 및 `Matrix2D` 셀 읽기·쓰기 구현
- [x] **원인별 입력·스키마 방어**: 모드 1 열 개수·비숫자 원인 구분 재입력 유도, 모드 2 비정상 키 예시·미로드 후보 목록 안내 및 케이스 단위 `FAIL` 격리
- [x] **정규화·허용오차·재검증**: `Cross`/`X` 표준 라벨 일원화, `abs(a - b) < 1e-9` 동점 판정 및 L1 가중치 정규화 보조 재검증 시뮬레이션 제공
- [x] **성능·병목 분석 및 리포트 동기화**: `repeats=1000` 명시 성능 표, 실측/이론 $O(N^2)$ 비율 분석, CPU 직렬 vs 스레드(GIL) 병목 실험 및 `report_state.json` 동기화 저장 완비
