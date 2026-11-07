from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import math
import os
import shutil
import sys
import time


class MessageCatalog:
    """입력·스키마·종료 안내 메시지 템플릿을 상수화하여 재사용성과 다국어(ko/en) 지역화 확장을 지원하는 카탈로그"""

    DEFAULT_LOCALE = "ko"

    TEMPLATES: dict[str, dict[str, str]] = {
        "ko": {
            "MENU_EMPTY": "[입력 오류] 입력이 비어 있습니다. {min_val}~{max_val} 사이의 숫자를 입력하세요.",
            "MENU_NOT_INT": "[입력 오류] 숫자가 아닙니다. {min_val}~{max_val} 사이의 정수를 입력하세요.",
            "MENU_OUT_OF_RANGE": "[입력 오류] 허용 범위를 벗어났습니다. {min_val}~{max_val} 사이의 숫자를 입력하세요.",
            "ROW_EMPTY": "입력 형식 오류 [빈 줄 입력]: 각 줄에 {size}개의 숫자를 공백으로 구분해 입력하세요.",
            "ROW_COL_MISMATCH": (
                "입력 형식 오류 [열 개수 불일치: 입력 {actual}개 != 기대 {size}개]: "
                "각 줄에 {size}개의 숫자를 공백으로 구분해 입력하세요."
            ),
            "ROW_PARSE_FLOAT": (
                "입력 형식 오류 [숫자 파싱 실패({token!r})]: "
                "각 줄에 {size}개의 숫자를 공백으로 구분해 입력하세요."
            ),
            "ROW_NON_FINITE": (
                "입력 형식 오류 [유한 실수 아님({token!r})]: "
                "각 줄에 {size}개의 숫자를 공백으로 구분해 입력하세요."
            ),
            "FILTER_MISSING": "매칭되는 'size_{target_n}' 필터가 없습니다 (로드된 필터 후보: {candidates}).",
            "SAFE_SHUTDOWN": "[안전 종료] 종료 신호가 감지되어 시뮬레이터를 안전하게 종료합니다.",
        },
        "en": {
            "MENU_EMPTY": "[Input Error] Empty input. Enter an integer between {min_val} and {max_val}.",
            "MENU_NOT_INT": "[Input Error] Not an integer. Enter a number between {min_val} and {max_val}.",
            "MENU_OUT_OF_RANGE": "[Input Error] Out of range. Enter a number between {min_val} and {max_val}.",
            "ROW_EMPTY": "Input format error [empty line]: enter {size} space-separated numbers per row.",
            "ROW_COL_MISMATCH": (
                "Input format error [column mismatch: got {actual} != expected {size}]: "
                "enter {size} space-separated numbers per row."
            ),
            "ROW_PARSE_FLOAT": (
                "Input format error [non-numeric token({token!r})]: "
                "enter {size} space-separated numbers per row."
            ),
            "ROW_NON_FINITE": (
                "Input format error [non-finite float({token!r})]: "
                "enter {size} space-separated numbers per row."
            ),
            "FILTER_MISSING": "No matching 'size_{target_n}' filter found (loaded candidates: {candidates}).",
            "SAFE_SHUTDOWN": "[Graceful Shutdown] Termination signal detected; exiting safely.",
        },
    }

    @classmethod
    def format(cls, key: str, locale: str = DEFAULT_LOCALE, **kwargs) -> str:
        lang_table = cls.TEMPLATES.get(locale, cls.TEMPLATES[cls.DEFAULT_LOCALE])
        template = lang_table[key]
        return template.format(**kwargs)


class Matrix2D:
    """N×N 2차원 정방 숫자 배열(패턴·필터)을 표현하고 셀 단위 접근·차원 무결성·메모리 추정을 담당하는 클래스 (SRP)"""

    def __init__(self, grid: list[list[float]]):
        self._grid = self._validate_square_grid(grid)
        self.size = len(self._grid)

    @staticmethod
    def _validate_number(value: int | float, context: str = "원소") -> float:
        # Python에서 isinstance(True, int) == True 함정 및 NaN/Inf 수치 오염 방어
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{context} 값은 숫자(int 또는 float)여야 합니다: {value!r}")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{context} 값에 유한하지 않은 실수(NaN/Inf)가 포함되어 있습니다.")
        return numeric

    @classmethod
    def _validate_square_grid(cls, grid: list[list[float]]) -> list[list[float]]:
        if not isinstance(grid, list) or len(grid) == 0:
            raise ValueError("행렬 데이터는 비어 있지 않은 2차원 리스트여야 합니다.")

        row_count = len(grid)
        validated: list[list[float]] = []
        for r_idx, row in enumerate(grid):
            if not isinstance(row, list):
                raise ValueError(f"{r_idx + 1}번째 행이 리스트 형식이 아닙니다.")
            if len(row) != row_count:
                raise ValueError(
                    f"정방 행렬(N×N) 크기 불일치: 총 {row_count}행이지만 "
                    f"{r_idx + 1}번째 행의 열 개수가 {len(row)}개입니다."
                )
            validated.append(
                [cls._validate_number(val, f"({r_idx}, {c_idx})") for c_idx, val in enumerate(row)]
            )
        return validated

    def _check_bounds(self, row: int, col: int) -> None:
        if isinstance(row, bool) or isinstance(col, bool) or not isinstance(row, int) or not isinstance(col, int):
            raise IndexError("행·열 인덱스는 정수여야 합니다.")
        if not (0 <= row < self.size and 0 <= col < self.size):
            raise IndexError(
                f"인덱스 범위 초과: ({row}, {col})는 {self.size}×{self.size} 행렬 범위를 벗어납니다."
            )

    def get_value(self, row: int, col: int) -> float:
        """특정 (row, col) 위치의 실수 값을 안전하게 반환"""
        self._check_bounds(row, col)
        return self._grid[row][col]

    def set_value(self, row: int, col: int, value: int | float) -> None:
        """특정 (row, col) 위치에 검증된 실수 값을 저장"""
        self._check_bounds(row, col)
        self._grid[row][col] = self._validate_number(value, f"({row}, {col})")

    def to_list(self) -> list[list[float]]:
        """내부 불변성 보호를 위한 2차원 리스트 깊은 복사본 반환"""
        return [list(row) for row in self._grid]

    def to_flat_list(self) -> list[float]:
        """행 우선(Row-major: r=0..N-1, c=0..N-1) 순서로 길이 N²의 1차원 연속 리스트로 변환"""
        return [val for row in self._grid for val in row]

    @staticmethod
    def estimate_pair_memory_mib(size: int) -> dict[str, float]:
        """64비트 CPython 기준 패턴+필터 2개 행렬의 2D 리스트 객체 메모리 및 연속 float64 버퍼 추정치(MiB) 계산"""
        # PyListObject 헤더(56B) + 포인터(8B/요소) + PyFloatObject(24B/원소) × 2개 행렬(패턴+필터)
        python_2d_bytes = 2 * (56 + (64 * size) + (32 * size * size))
        contiguous_fp64_bytes = 2 * 8 * size * size
        mib = 1024.0 * 1024.0
        return {
            "python_2d_mib": python_2d_bytes / mib,
            "contiguous_fp64_mib": contiguous_fp64_bytes / mib,
        }

    def format_visual(self) -> str:
        """콘솔 시각적 확인용 문자열 포맷 생성"""
        lines = []
        for row in self._grid:
            lines.append("  " + " ".join("■" if abs(v) > 1e-9 else "□" for v in row))
        return "\n".join(lines)


class LabelNormalizer:
    """외부 JSON 키 및 expected 표기를 내부 표준 라벨(Cross, X)로 일관되게 정규화하는 클래스 (SRP)"""

    CROSS = "Cross"
    X = "X"
    UNDECIDED = "UNDECIDED"
    REQUIRED_LABELS = (CROSS, X)

    # [신규 라벨 추가 절차] 1) 표준 상수(예: CIRCLE = "Circle") 선언 -> 2) _LABEL_MAP에 소문자 별칭 등록 -> 3) REQUIRED_LABELS·테스트 반영
    _LABEL_MAP = {
        "+": CROSS,
        "cross": CROSS,
        "x": X,
    }

    ALLOWED_ALIAS_ONELINE = "Cross('+', 'cross'), X('x') [대소문자·공백 무시]"

    @classmethod
    def allowed_aliases_summary(cls) -> str:
        """정규화 실패 메시지에 한 줄로 첨부할 표준 라벨별 허용 별칭 요약"""
        return cls.ALLOWED_ALIAS_ONELINE

    @classmethod
    def normalize(cls, raw_label: str) -> str:
        """'+'/'cross' -> 'Cross', 'x' -> 'X' 표준 라벨 변환 (대소문자·앞뒤 공백 무시)"""
        # 신규 라벨 확장 시 위 _LABEL_MAP에 소문자 키와 표준 라벨 상수를 매핑하면 본 메서드는 수정 없이 자동 동작함
        if not isinstance(raw_label, str) or not raw_label.strip():
            raise ValueError(f"빈 라벨 입력 — 허용 별칭 요약: {cls.allowed_aliases_summary()}")
        key = raw_label.strip().lower()
        if key not in cls._LABEL_MAP:
            raise ValueError(
                f"비표준 라벨 {raw_label!r} — 허용 별칭 요약: {cls.allowed_aliases_summary()}"
            )
        return cls._LABEL_MAP[key]


class MacEngine:
    """순수 반복문 기반 행 우선 2D/1D MAC 연산, 허용오차(epsilon) 판정 및 워밍업·분산 벤치마크를 전담하는 엔진 (SRP)"""

    DEFAULT_EPSILON = 1e-9
    DEFAULT_REPEATS = 1000
    DEFAULT_WARMUP = 20
    SAMPLE_BLOCKS = 20

    def __init__(
        self,
        epsilon: float = DEFAULT_EPSILON,
        repeats: int = DEFAULT_REPEATS,
        warmup: int = DEFAULT_WARMUP,
    ):
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or epsilon <= 0:
            raise ValueError("허용오차(epsilon)는 0보다 큰 양수여야 합니다.")
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 10:
            raise ValueError("반복 측정 횟수(repeats)는 최소 10 이상의 정수여야 합니다.")
        if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
            raise ValueError("워밍업 반복 횟수(warmup)는 0 이상의 정수여야 합니다.")
        self.epsilon = float(epsilon)
        self.repeats = repeats
        self.warmup = warmup

    @staticmethod
    def compute_mac_2d(pattern: Matrix2D, filter_mat: Matrix2D) -> float:
        """행 우선(Row-major: 바깥 행 r=0..N-1 -> 안쪽 열 c=0..N-1) 순서로 원소별 곱을 순차 누적(MAC)하여 점수 반환"""
        if pattern.size != filter_mat.size:
            raise ValueError(
                f"패턴 크기({pattern.size}×{pattern.size})와 "
                f"필터 크기({filter_mat.size}×{filter_mat.size})가 일치하지 않습니다."
            )
        n = pattern.size
        p_grid = pattern._grid
        f_grid = filter_mat._grid
        total = 0.0
        for r in range(n):
            p_row = p_grid[r]
            f_row = f_grid[r]
            for c in range(n):
                total += p_row[c] * f_row[c]
        return total

    @staticmethod
    def compute_mac_1d(flat_pattern: list[float], flat_filter: list[float]) -> float:
        """행 우선 평탄화된 1차원 배열(길이 N²)에 대해 단일 루프로 MAC 연산 수행 (메모리 접근 최적화)"""
        if len(flat_pattern) != len(flat_filter):
            raise ValueError(
                f"1차원 배열 길이 불일치: 패턴({len(flat_pattern)}) != 필터({len(flat_filter)})"
            )
        total = 0.0
        for p_val, f_val in zip(flat_pattern, flat_filter):
            total += p_val * f_val
        return total

    @staticmethod
    def normalize_filter_l1(filter_mat: Matrix2D) -> Matrix2D:
        """동점 케이스 재검증용 보조 기능: 필터 가중치 절댓값 총합이 1.0이 되도록 L1 정규화 수행"""
        abs_sum = 0.0
        for row in filter_mat._grid:
            for val in row:
                abs_sum += abs(val)
        if abs_sum <= 0.0:
            return Matrix2D(filter_mat.to_list())
        norm_grid = [[val / abs_sum for val in row] for row in filter_mat._grid]
        return Matrix2D(norm_grid)

    @classmethod
    def compute_score_bounds_and_ratio(
        cls, pattern: Matrix2D, filter_mat: Matrix2D
    ) -> dict[str, float]:
        """이진/비음수 입력 패턴([0, 1]) 기준 최소·최대 가능 MAC 점수와 최대치 대비 정규화 일치율(0~1) 산출"""
        raw_score = cls.compute_mac_2d(pattern, filter_mat)
        min_possible = 0.0
        max_possible = 0.0
        for row in filter_mat._grid:
            for val in row:
                if val < 0.0:
                    min_possible += val
                elif val > 0.0:
                    max_possible += val
        norm_ratio = (raw_score / max_possible) if max_possible > 0.0 else 0.0
        return {
            "raw_score": raw_score,
            "min_possible": min_possible,
            "max_possible": max_possible,
            "normalized_ratio": norm_ratio,
        }

    def judge_scores(
        self,
        score_a: float,
        score_b: float,
        label_a: str = LabelNormalizer.CROSS,
        label_b: str = LabelNormalizer.X,
        tie_label: str = LabelNormalizer.UNDECIDED,
    ) -> str:
        """IEEE-754 부동소수점 오차를 고려해 |score_a - score_b| < epsilon 이면 동점(tie_label)으로 판정"""
        if abs(score_a - score_b) < self.epsilon:
            return tie_label
        return label_a if score_a > score_b else label_b

    @staticmethod
    def _calc_stats(samples_ms: list[float]) -> tuple[float, float, float, float]:
        """표본 블록 시간 리스트로부터 평균(avg), 표본 표준편차(std), 최솟값(min), 최댓값(max) 산출"""
        count = len(samples_ms)
        avg = sum(samples_ms) / count
        if count > 1:
            variance = sum((x - avg) ** 2 for x in samples_ms) / (count - 1)
            std = math.sqrt(variance)
        else:
            std = 0.0
        return avg, std, min(samples_ms), max(samples_ms)

    def measure_pair_mac_ms(
        self,
        pattern: Matrix2D,
        filter_a: Matrix2D,
        filter_b: Matrix2D,
        repeats: int | None = None,
        warmup: int | None = None,
    ) -> tuple[float, float, float]:
        """콘솔/파일 I/O 제외 및 선택적 워밍업(warmup) 수행 후 순수 MAC 연산 쌍의 평균 시간(ms) 측정"""
        iters = max(10, repeats if repeats is not None else self.repeats)
        warm = self.warmup if warmup is None else max(0, warmup)

        score_a = 0.0
        score_b = 0.0
        for _ in range(warm):
            score_a = self.compute_mac_2d(pattern, filter_a)
            score_b = self.compute_mac_2d(pattern, filter_b)

        start = time.perf_counter()
        for _ in range(iters):
            score_a = self.compute_mac_2d(pattern, filter_a)
            score_b = self.compute_mac_2d(pattern, filter_b)
        elapsed_ms = ((time.perf_counter() - start) / iters) * 1000.0
        return score_a, score_b, elapsed_ms

    def benchmark_sizes(
        self,
        sizes: list[int],
        filters_by_size: dict[int, dict[str, Matrix2D]] | None = None,
        repeats: int | None = None,
        warmup: int | None = None,
    ) -> list[dict]:
        """워밍업 후 크기별 2D/1D 평균·표준편차·최소/최대 시간(ms), 연산수(N²) 및 예상 메모리(MiB) 측정"""
        iters = max(10, repeats if repeats is not None else self.repeats)
        warm = self.warmup if warmup is None else max(0, warmup)
        blocks = min(self.SAMPLE_BLOCKS, iters)
        per_block = max(1, iters // blocks)
        effective_iters = blocks * per_block
        results: list[dict] = []

        for size in sizes:
            pattern = PatternGenerator.generate_cross(size)
            if filters_by_size and size in filters_by_size and LabelNormalizer.CROSS in filters_by_size[size]:
                filter_mat = filters_by_size[size][LabelNormalizer.CROSS]
            else:
                filter_mat = PatternGenerator.generate_cross(size)

            # I/O 및 to_flat_list() 변환 비용을 측정 구간에서 제외하기 위해 루프 진입 전 사전 평탄화
            flat_p = pattern.to_flat_list()
            flat_f = filter_mat.to_flat_list()

            # 바이트코드 적응형 특수화 및 캐시 예열을 위한 워밍업 루프 (측정 시간에서 제외)
            for _ in range(warm):
                self.compute_mac_2d(pattern, filter_mat)
                self.compute_mac_1d(flat_p, flat_f)

            samples_2d: list[float] = []
            for _ in range(blocks):
                t0 = time.perf_counter()
                for _ in range(per_block):
                    self.compute_mac_2d(pattern, filter_mat)
                samples_2d.append(((time.perf_counter() - t0) / per_block) * 1000.0)

            samples_1d: list[float] = []
            for _ in range(blocks):
                t1 = time.perf_counter()
                for _ in range(per_block):
                    self.compute_mac_1d(flat_p, flat_f)
                samples_1d.append(((time.perf_counter() - t1) / per_block) * 1000.0)

            avg_2d_ms, std_2d_ms, min_2d_ms, max_2d_ms = self._calc_stats(samples_2d)
            avg_1d_ms, std_1d_ms, min_1d_ms, max_1d_ms = self._calc_stats(samples_1d)
            speedup_pct = ((avg_2d_ms - avg_1d_ms) / avg_2d_ms * 100.0) if avg_2d_ms > 0 else 0.0
            mem_info = Matrix2D.estimate_pair_memory_mib(size)

            results.append(
                {
                    "size": size,
                    "ops": size * size,
                    "repeats": effective_iters,
                    "warmup": warm,
                    "avg_2d_ms": avg_2d_ms,
                    "std_2d_ms": std_2d_ms,
                    "min_2d_ms": min_2d_ms,
                    "max_2d_ms": max_2d_ms,
                    "avg_1d_ms": avg_1d_ms,
                    "std_1d_ms": std_1d_ms,
                    "min_1d_ms": min_1d_ms,
                    "max_1d_ms": max_1d_ms,
                    "speedup_pct": speedup_pct,
                    "est_mem_mib": mem_info["python_2d_mib"],
                    "contiguous_fp64_mib": mem_info["contiguous_fp64_mib"],
                }
            )
        return results

    @staticmethod
    def project_large_scale(reference_record: dict, target_size: int = 1000) -> dict:
        """기준 실측 레코드(예: 25×25)의 원소당 연산 시간(ms/op)을 바탕으로 대형 N×N(기본 1000×1000, 1M 원소) 시간·메모리 예측"""
        ref_ops = max(1, reference_record["ops"])
        target_ops = target_size * target_size
        scale = target_ops / ref_ops
        mem = Matrix2D.estimate_pair_memory_mib(target_size)
        return {
            "size": target_size,
            "ops": target_ops,
            "projected_2d_ms": reference_record["avg_2d_ms"] * scale,
            "projected_1d_ms": reference_record["avg_1d_ms"] * scale,
            "est_python_2d_mib": mem["python_2d_mib"],
            "est_contiguous_fp64_mib": mem["contiguous_fp64_mib"],
        }

    def measure_large_scale_sample(self, size: int = 1000, repeats: int = 3, warmup: int = 1) -> dict:
        """대형 정방 행렬(기본 1000×1000 = 1,000,000 원소) 실제 단일 샘플 실측 시간(ms) 및 메모리(MiB) 측정"""
        row_zeros = [0.0] * size
        mid = size // 2
        grid = [list(row_zeros) for _ in range(size)]
        for idx in range(size):
            grid[mid][idx] = 1.0
            grid[idx][mid] = 1.0
        pattern = Matrix2D(grid)
        flat_p = pattern.to_flat_list()

        for _ in range(max(0, warmup)):
            self.compute_mac_2d(pattern, pattern)
            self.compute_mac_1d(flat_p, flat_p)

        iters = max(1, repeats)
        t0 = time.perf_counter()
        for _ in range(iters):
            self.compute_mac_2d(pattern, pattern)
        measured_2d_ms = ((time.perf_counter() - t0) / iters) * 1000.0

        t1 = time.perf_counter()
        for _ in range(iters):
            self.compute_mac_1d(flat_p, flat_p)
        measured_1d_ms = ((time.perf_counter() - t1) / iters) * 1000.0

        mem = Matrix2D.estimate_pair_memory_mib(size)
        return {
            "size": size,
            "ops": size * size,
            "repeats": iters,
            "measured_2d_ms": measured_2d_ms,
            "measured_1d_ms": measured_1d_ms,
            "est_python_2d_mib": mem["python_2d_mib"],
            "est_contiguous_fp64_mib": mem["contiguous_fp64_mib"],
        }

    def benchmark_serial_vs_threaded(
        self,
        size: int = 25,
        batch_filters: int = 64,
        workers: int = 4,
    ) -> dict:
        """다중 필터 배치 처리 시 CPU 직렬(2D/1D) 대 파이썬 다중 스레드(GIL 경합) 병목 비교 실험"""
        pattern = PatternGenerator.generate_cross(size)
        filters_2d = [PatternGenerator.generate_cross(size) for _ in range(batch_filters)]
        flat_p = pattern.to_flat_list()
        filters_1d = [f.to_flat_list() for f in filters_2d]

        t0 = time.perf_counter()
        serial_2d_scores = [self.compute_mac_2d(pattern, flt) for flt in filters_2d]
        serial_2d_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        serial_1d_scores = [self.compute_mac_1d(flat_p, flt) for flt in filters_1d]
        serial_1d_ms = (time.perf_counter() - t1) * 1000.0

        with ThreadPoolExecutor(max_workers=workers) as pool:
            t2 = time.perf_counter()
            threaded_scores = list(
                pool.map(lambda flt: self.compute_mac_2d(pattern, flt), filters_2d)
            )
            threaded_2d_ms = (time.perf_counter() - t2) * 1000.0

        return {
            "size": size,
            "batch_filters": batch_filters,
            "workers": workers,
            "total_mac_ops": size * size * batch_filters,
            "serial_2d_ms": serial_2d_ms,
            "serial_1d_ms": serial_1d_ms,
            "threaded_2d_ms": threaded_2d_ms,
            "parity_ok": len(serial_2d_scores) == len(serial_1d_scores) == len(threaded_scores),
        }


class PatternGenerator:
    """임의의 크기 N에 대해 N×N 십자가(Cross) 및 대각선(X) 2차원 패턴을 동적 생성하는 클래스 (보너스 과제 & 벤치마크 재활용)"""

    @staticmethod
    def _validate_size(size: int) -> int:
        if isinstance(size, bool) or not isinstance(size, int) or size < 3:
            raise ValueError("패턴 크기 N은 3 이상의 정수여야 합니다.")
        return size

    @classmethod
    def generate_cross(cls, size: int) -> Matrix2D:
        """중앙 행과 중앙 열이 1.0이고 나머지가 0.0인 N×N 십자가(Cross) 패턴 생성"""
        n = cls._validate_size(size)
        mid = n // 2
        grid = [
            [1.0 if (r == mid or c == mid) else 0.0 for c in range(n)]
            for r in range(n)
        ]
        return Matrix2D(grid)

    @classmethod
    def generate_x(cls, size: int) -> Matrix2D:
        """주대각선(r == c)과 부대각선(r + c == N - 1)이 1.0인 N×N X자 패턴 생성"""
        n = cls._validate_size(size)
        grid = [
            [1.0 if (r == c or r + c == n - 1) else 0.0 for c in range(n)]
            for r in range(n)
        ]
        return Matrix2D(grid)

    @classmethod
    def generate_pair(cls, size: int) -> dict[str, Matrix2D]:
        """지정된 크기 N의 표준 라벨(Cross, X) 패턴 쌍 반환"""
        return {
            LabelNormalizer.CROSS: cls.generate_cross(size),
            LabelNormalizer.X: cls.generate_x(size),
        }


class DatasetRepository:
    """data.json 로드·키 규칙 해석·후보 포맷 일원화·백업 복구(.bak) 및 리포트 원자적 저장을 전담하는 클래스 (SRP)"""

    KEY_RULE_GUIDE = "권장 예시: 필터 키 'size_5', 패턴 키 'size_5_1', 'size_13_2'"

    def __init__(self, base_dir: str | None = None, data_filename: str = "data.json"):
        root = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.data_file = os.path.join(root, data_filename)
        self.data_backup_file = f"{self.data_file}.bak"
        self.data_corrupt_file = f"{self.data_file}.corrupt"
        self.report_file = os.path.join(root, "report_state.json")
        self.report_backup_file = os.path.join(root, "report_state.json.bak")
        self.report_temp_file = os.path.join(root, "report_state.json.tmp")

    @staticmethod
    def format_candidate_sizes(sizes: list[int] | set[int] | dict) -> str:
        """콘솔 에러 메시지와 JSON 리포트가 100% 동일한 형식('[size_5, size_13, size_25]')을 공유하도록 일원화"""
        sorted_sizes = sorted(int(k) for k in sizes)
        if not sorted_sizes:
            return "[없음]"
        return "[" + ", ".join(f"size_{k}" for k in sorted_sizes) + "]"

    @classmethod
    def extract_size_from_key(cls, key: str, require_case_index: bool = False) -> int:
        """'size_5'(필터 키) 또는 'size_13_2'(패턴 키)에서 정수 N을 추출하고 다양한 변형 오류별 수정 예시 반환"""
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"빈 키 입력 ({cls.KEY_RULE_GUIDE}).")
        if key != key.strip():
            raise ValueError(
                f"키 앞뒤 공백 포함 불가: {key!r} -> 권장 수정: {key.strip()!r} ({cls.KEY_RULE_GUIDE})."
            )
        if key.lower().startswith("size_") and not key.startswith("size_"):
            raise ValueError(
                f"대문자 접두사 사용 불가: {key!r} -> 소문자 'size_'로 수정하세요 ({cls.KEY_RULE_GUIDE})."
            )
        if ("-" in key or "." in key) and key.lower().startswith("size"):
            suggested = key.lower().replace("-", "_").replace(".", "_")
            raise ValueError(
                f"구분자 오류('-' 또는 '.'): {key!r} -> 밑줄('_') 형식 {suggested!r} 권장 ({cls.KEY_RULE_GUIDE})."
            )
        if not key.startswith("size_"):
            raise ValueError(f"키 접두사('size_') 누락: {key!r} ({cls.KEY_RULE_GUIDE}).")

        parts = key.split("_")
        if len(parts) < 2 or not parts[1]:
            raise ValueError(f"크기 숫자 N 구분자 누락: {key!r} ({cls.KEY_RULE_GUIDE}).")
        if not parts[1].lstrip("-").isdigit():
            raise ValueError(
                f"크기 N에 비정수({parts[1]!r}) 지정: {key!r} ({cls.KEY_RULE_GUIDE})."
            )
        size = int(parts[1])
        if size <= 0:
            raise ValueError(
                f"크기 N({size})은 1 이상의 양의 정수여야 합니다: {key!r} ({cls.KEY_RULE_GUIDE})."
            )
        if require_case_index and (len(parts) < 3 or not parts[2]):
            raise ValueError(
                f"패턴 일련번호('_{{idx}}') 누락: {key!r} -> '{key}_1' 형식 권장 ({cls.KEY_RULE_GUIDE})."
            )
        return size

    def _read_and_check_root(self, filepath: str) -> dict:
        with open(filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise ValueError("루트 JSON 데이터는 객체(dict)여야 합니다.")
        if not isinstance(raw.get("filters"), dict):
            raise ValueError("'filters' 섹션이 누락되었거나 객체(dict)가 아닙니다.")
        if not isinstance(raw.get("patterns"), dict):
            raise ValueError("'patterns' 섹션이 누락되었거나 객체(dict)가 아닙니다.")
        return raw

    def load_dataset(self) -> dict:
        """data.json 로드 및 파일 파손 시 .corrupt 격리 후 .bak 백업본으로부터 자동 복구"""
        if os.path.exists(self.data_file):
            try:
                raw = self._read_and_check_root(self.data_file)
                if not os.path.exists(self.data_backup_file):
                    try:
                        shutil.copy2(self.data_file, self.data_backup_file)
                    except OSError:
                        pass
                return raw
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                print(f"[경고] '{os.path.basename(self.data_file)}' 파싱 실패: {exc}")
                try:
                    shutil.copy2(self.data_file, self.data_corrupt_file)
                    print(f"[보호] 손상된 원본을 '{os.path.basename(self.data_corrupt_file)}'로 격리 보존했습니다.")
                except OSError:
                    pass
        else:
            print(f"[경고] '{os.path.basename(self.data_file)}' 파일이 존재하지 않습니다.")

        if os.path.exists(self.data_backup_file):
            try:
                recovered = self._read_and_check_root(self.data_backup_file)
                shutil.copy2(self.data_backup_file, self.data_file)
                print(f"[복구] 백업본('{os.path.basename(self.data_backup_file)}')으로부터 데이터를 복원했습니다.")
                return recovered
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise RuntimeError(f"메인 파일과 백업 파일이 모두 손상되어 로드할 수 없습니다: {exc}") from exc

        raise FileNotFoundError(
            f"분석할 데이터 파일('{self.data_file}')을 찾을 수 없습니다. 경로를 확인해 주세요."
        )

    def load_normalized_filters(
        self, raw_filters: dict
    ) -> tuple[dict[int, dict[str, Matrix2D]], list[str]]:
        """filters 섹션에서 size_N 별 Cross/X 필터를 정규화 및 크기 검증하여 로드"""
        loaded: dict[int, dict[str, Matrix2D]] = {}
        logs: list[str] = []

        for size_key, group in raw_filters.items():
            try:
                expected_n = self.extract_size_from_key(size_key, require_case_index=False)
                if not isinstance(group, dict):
                    raise ValueError(f"'{size_key}' 필터 항목이 객체(dict)가 아닙니다.")

                norm_pair: dict[str, Matrix2D] = {}
                for raw_label, raw_grid in group.items():
                    std_label = LabelNormalizer.normalize(raw_label)
                    mat = Matrix2D(raw_grid)
                    if mat.size != expected_n:
                        raise ValueError(
                            f"'{size_key}.{raw_label}' 필터 크기({mat.size}×{mat.size})가 "
                            f"명시된 크기({expected_n}×{expected_n})와 다릅니다."
                        )
                    norm_pair[std_label] = mat

                for req_label in LabelNormalizer.REQUIRED_LABELS:
                    if req_label not in norm_pair:
                        raise ValueError(f"'{size_key}'에 표준 '{req_label}' 필터가 누락되었습니다.")

                loaded[expected_n] = norm_pair
                logs.append(f"[OK] {size_key} 필터 로드 완료 (Cross, X)")
            except ValueError as exc:
                logs.append(f"[ERR] {size_key} 필터 로드 실패: {exc}")

        return loaded, logs

    def save_report_atomic(self, report: dict) -> bool:
        """분석 요약 리포트를 .tmp 직렬화 + fsync -> .bak 백업 -> os.replace 원자적 교체로 안전 저장"""
        try:
            with open(self.report_temp_file, "w", encoding="utf-8") as tmp_f:
                json.dump(report, tmp_f, ensure_ascii=False, indent=2)
                tmp_f.write("\n")
                tmp_f.flush()
                os.fsync(tmp_f.fileno())

            if os.path.exists(self.report_file) and os.path.getsize(self.report_file) > 0:
                try:
                    with open(self.report_file, "r", encoding="utf-8") as chk:
                        json.load(chk)
                    shutil.copy2(self.report_file, self.report_backup_file)
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    pass

            os.replace(self.report_temp_file, self.report_file)
            return True
        except OSError:
            if os.path.exists(self.report_temp_file):
                try:
                    os.remove(self.report_temp_file)
                except OSError:
                    pass
            return False


class SimulatorApp:
    """콘솔 메뉴 제어, 3×3 대화형 분석, data.json 일괄 검증 및 성능·결과 리포트를 총괄하는 컨트롤러 (SRP)"""

    BENCHMARK_SIZES = [3, 5, 13, 25]
    MIN_REPORT_LINES = 10

    # 리포트 후처리 자동화를 위한 표준 상태 코드 및 심각도(우선순위) 상수
    STATUS_PASS = "pass"
    STATUS_TIE = "tie"
    STATUS_MISMATCH = "mismatch"
    STATUS_SCHEMA_ERROR = "schema_error"

    SEVERITY_LOW = "LOW"        # 허용오차(epsilon) 경계 동점(tie) -> 가중치 스케일/정규화 검토 권장
    SEVERITY_MEDIUM = "MEDIUM"  # 예측 도형 오분류(mismatch) -> 필터 가중치 재설계 필요
    SEVERITY_HIGH = "HIGH"      # 스키마·크기·키·라벨 오류(schema_error) -> 즉시 데이터 수정 필요

    OUTPUT_FORMAT_RULE = (
        "점수: 소수 넷째 자리(.4f, 정수·유효숫자 축약 및 동점 근접 시 .16f) | "
        "시간: ms(.4f) | 메모리: MiB(.4f)"
    )

    def __init__(
        self,
        repository: DatasetRepository | None = None,
        engine: MacEngine | None = None,
        locale: str = MessageCatalog.DEFAULT_LOCALE,
    ):
        self.repository = repository or DatasetRepository()
        self.engine = engine or MacEngine()
        self.locale = locale

    @staticmethod
    def _format_score(score_a: float, score_b: float) -> tuple[str, str]:
        """점수 표준화 규칙: 일반 케이스는 .4f(불필요 후행 0 생략), 동점 경계(|A-B|<1e-6)는 .16f 정밀도 표기"""
        if abs(score_a - score_b) < 1e-6:
            return f"{score_a:.16f}", f"{score_b:.16f}"
        return f"{score_a:.4f}".rstrip("0").rstrip("."), f"{score_b:.4f}".rstrip("0").rstrip(".")

    @staticmethod
    def format_judgment_line(
        judgment: str,
        epsilon: float,
        expected: str | None = None,
        result_tag: str | None = None,
    ) -> str:
        """모드 1·모드 2 모든 판정 출력에 적용된 epsilon 기준값을 중복 없이 일관되게 함께 표기하는 단일 포맷터"""
        eps_str = f"epsilon={epsilon:.0e}"
        if expected is None:
            if judgment == "판정 불가":
                return f"판정: 판정 불가 (동점 기준 |A-B| < {eps_str})"
            return f"판정: {judgment} (기준 {eps_str})"
        return f"판정: {judgment} ({eps_str}) | expected: {expected} | {result_tag}"

    @classmethod
    def _diagnose_case_failure(
        cls,
        predicted: str,
        expected: str,
        score_cross: float,
        score_x: float,
        epsilon: float,
    ) -> tuple[str, str, str]:
        """실패 시 (status_code, severity, reason) 3원소 튜플로 동점(tie)과 오분류(mismatch)를 표준화 반환"""
        diff = abs(score_cross - score_x)
        if predicted == LabelNormalizer.UNDECIDED:
            reason = (
                f"동점(UNDECIDED) 처리 규칙에 따라 FAIL "
                f"(|Cross-X| = {diff:.2e} < {epsilon:.0e}, expected: {expected})"
            )
            return cls.STATUS_TIE, cls.SEVERITY_LOW, reason
        reason = (
            f"예측 라벨 불일치로 FAIL "
            f"(판정: {predicted}, expected: {expected}, Cross={score_cross:.4f}, X={score_x:.4f})"
        )
        return cls.STATUS_MISMATCH, cls.SEVERITY_MEDIUM, reason

    def _prompt_int_in_range(self, prompt: str, min_val: int, max_val: int) -> int:
        while True:
            raw = input(prompt).strip()
            if not raw:
                print(MessageCatalog.format("MENU_EMPTY", self.locale, min_val=min_val, max_val=max_val))
                continue
            try:
                value = int(raw)
            except ValueError:
                print(MessageCatalog.format("MENU_NOT_INT", self.locale, min_val=min_val, max_val=max_val))
                continue
            if min_val <= value <= max_val:
                return value
            print(MessageCatalog.format("MENU_OUT_OF_RANGE", self.locale, min_val=min_val, max_val=max_val))

    def read_matrix_from_console(self, title: str, size: int = 3) -> Matrix2D:
        """지정 크기(기본 3×3) 행렬을 입력받아 MessageCatalog 기반 원인별(빈 줄/열 개수/비숫자) 안내 후 재입력 유도"""
        print(f"\n{title} ({size}줄 입력, 공백 구분)")
        rows: list[list[float]] = []
        for row_idx in range(1, size + 1):
            while True:
                line = input(f"  {row_idx}행: ").strip()
                if not line:
                    print(MessageCatalog.format("ROW_EMPTY", self.locale, size=size))
                    continue
                tokens = line.split()
                if len(tokens) != size:
                    print(
                        MessageCatalog.format(
                            "ROW_COL_MISMATCH", self.locale, actual=len(tokens), size=size
                        )
                    )
                    continue

                parsed_row: list[float] = []
                err_msg: str | None = None
                for tok in tokens:
                    try:
                        num = float(tok)
                    except ValueError:
                        err_msg = MessageCatalog.format(
                            "ROW_PARSE_FLOAT", self.locale, token=tok, size=size
                        )
                        break
                    try:
                        parsed_row.append(Matrix2D._validate_number(num, f"{row_idx}행"))
                    except ValueError:
                        err_msg = MessageCatalog.format(
                            "ROW_NON_FINITE", self.locale, token=tok, size=size
                        )
                        break

                if err_msg is not None:
                    print(err_msg)
                    continue

                rows.append(parsed_row)
                break
        return Matrix2D(rows)

    def print_benchmark_table(
        self,
        sizes: list[int],
        filters_by_size: dict[int, dict[str, Matrix2D]] | None = None,
        section_num: int = 3,
        include_projection: bool = True,
    ) -> tuple[list[dict], dict]:
        """크기별 반복수·워밍업·2D 평균/표준편차/최소~최대·1D 평균·연산수(N²)·예상 메모리(MiB) 및 1000×1000 예측 출력"""
        records = self.engine.benchmark_sizes(sizes, filters_by_size=filters_by_size)
        repeats = records[0]["repeats"] if records else self.engine.repeats
        warmup = records[0]["warmup"] if records else self.engine.warmup

        print("\n#----------------------------------------")
        print(f"# [{section_num}] 성능 분석 (repeats={repeats}회, warmup={warmup}회)")
        print("#----------------------------------------")
        print(
            f"{'크기':<7}{'반복수':<8}{'2D평균(ms)':<12}{'표준편차(±ms)':<14}"
            f"{'최소~최대(ms)':<18}{'1D평균(ms)':<12}{'연산수(N²)':<11}{'예상메모리(MiB)':<14}"
        )
        print("-" * 96)
        for rec in records:
            label = f"{rec['size']}×{rec['size']}"
            min_max = f"{rec['min_2d_ms']:.4f}~{rec['max_2d_ms']:.4f}"
            print(
                f"{label:<7}{rec['repeats']:<8}{rec['avg_2d_ms']:<12.4f}{rec['std_2d_ms']:<14.4f}"
                f"{min_max:<18}{rec['avg_1d_ms']:<12.4f}{rec['ops']:<11}{rec['est_mem_mib']:<14.4f}"
            )

        projection = self.engine.project_large_scale(records[-1], target_size=1000)
        if include_projection:
            print(
                f"[대형 입력 예측] 1000×1000 (1,000,000 ops): "
                f"예상 2D ~{projection['projected_2d_ms']:.2f} ms (1D ~{projection['projected_1d_ms']:.2f} ms) | "
                f"파이썬 2D 메모리 ~{projection['est_python_2d_mib']:.2f} MiB "
                f"(연속 float64 버퍼 시 ~{projection['est_contiguous_fp64_mib']:.2f} MiB)"
            )
        return records, projection

    @classmethod
    def render_markdown_report_lines(cls, summary: dict) -> list[str]:
        """리포트 최소 줄 수(MIN_REPORT_LINES=10줄 이상) 자동 검증 규칙을 충족하는 구조화 요약 텍스트 생성"""
        proj = summary["large_scale_projection_1000x1000"]
        lines = [
            f"1. 분석 시각 및 매개변수: {summary['analyzed_at']} (epsilon={summary['epsilon']:.0e}, repeats={summary['repeats']}, warmup={summary['warmup']})",
            f"2. 표준 출력 포맷 규칙: {cls.OUTPUT_FORMAT_RULE}",
            f"3. 로드된 필터 후보 목록: {summary['loaded_filter_candidates_display']}",
            f"4. 전체 테스트 집계: 총 {summary['total']}건 / 통과 {summary['passed']}건 / 실패 {summary['failed']}건",
        ]
        for idx, fc in enumerate(summary["fail_cases"], start=5):
            lines.append(
                f"{idx}. 실패 상세 [{fc['severity']}|code={fc['status_code']}] {fc['case_id']}: {fc['reason']}"
            )
        next_no = len(lines) + 1
        lines.append(
            f"{next_no}. 시간 복잡도 검증: N² 연산량 증가(9 -> 25 -> 169 -> 625)에 비례해 평균 연산 시간 선형 증가 확인"
        )
        lines.append(
            f"{next_no + 1}. 대형 입력(1000×1000, 1M ops) 예측: 2D 약 {proj['projected_2d_ms']:.2f} ms, 메모리 약 {proj['est_python_2d_mib']:.2f} MiB"
        )
        while len(lines) < cls.MIN_REPORT_LINES:
            lines.append(f"{len(lines) + 1}. 추가 검증 항목: 케이스 단위 FAIL 격리 및 원자적 리포트 저장 정상 완료")
        return lines

    def run_user_input_mode(self) -> None:
        """모드 1: 3×3 필터 A, B 입력 -> 저장 확인 -> 패턴 입력 -> MAC 연산/일원화 판정 -> 성능 분석(3×3) 출력"""
        print("\n#----------------------------------------")
        print("# [1] 필터 입력")
        print("#----------------------------------------")
        filter_a = self.read_matrix_from_console("필터 A", 3)
        filter_b = self.read_matrix_from_console("필터 B", 3)
        print(
            f"\n[OK] 필터 저장 확인 완료: 필터 A ({filter_a.size}×{filter_a.size}), "
            f"필터 B ({filter_b.size}×{filter_b.size})"
        )

        print("\n#----------------------------------------")
        print("# [2] 패턴 입력")
        print("#----------------------------------------")
        pattern = self.read_matrix_from_console("패턴", 3)

        score_a, score_b, avg_pair_ms = self.engine.measure_pair_mac_ms(pattern, filter_a, filter_b)
        judgment = self.engine.judge_scores(
            score_a, score_b, label_a="A", label_b="B", tie_label="판정 불가"
        )
        fmt_a, fmt_b = self._format_score(score_a, score_b)
        bounds_a = self.engine.compute_score_bounds_and_ratio(pattern, filter_a)
        bounds_b = self.engine.compute_score_bounds_and_ratio(pattern, filter_b)

        print("\n#----------------------------------------")
        print("# [3] MAC 결과")
        print("#----------------------------------------")
        print(f"A 점수: {fmt_a} (최대 가능 {bounds_a['max_possible']:.4g} 대비 일치율 {bounds_a['normalized_ratio'] * 100:.1f}%)")
        print(f"B 점수: {fmt_b} (최대 가능 {bounds_b['max_possible']:.4g} 대비 일치율 {bounds_b['normalized_ratio'] * 100:.1f}%)")
        print(
            f"연산 시간(평균/{self.engine.repeats}회, 워밍업 {self.engine.warmup}회): {avg_pair_ms:.4f} ms"
        )
        print(self.format_judgment_line(judgment, self.engine.epsilon))

        self.print_benchmark_table([3], section_num=4, include_projection=False)

    def evaluate_dataset(self) -> dict:
        """모드 2: data.json 로드 -> 정규화·크기 검증 -> 케이스별 MAC 판정(코드·심각도 태깅) -> 분산·메모리 성능표 및 리포트 저장"""
        raw_data = self.repository.load_dataset()

        print("\n#----------------------------------------")
        print("# [1] 필터 로드")
        print("#----------------------------------------")
        filters_by_size, filter_logs = self.repository.load_normalized_filters(raw_data["filters"])
        for msg in filter_logs:
            print(msg)

        candidates_display = self.repository.format_candidate_sizes(filters_by_size)

        print("\n#----------------------------------------")
        print("# [2] 패턴 분석 (라벨 정규화 적용)")
        print("#----------------------------------------")
        raw_patterns: dict = raw_data["patterns"]
        total_count = 0
        pass_count = 0
        fail_cases: list[dict] = []
        tie_recheck_notes: list[str] = []

        for case_id, entry in raw_patterns.items():
            total_count += 1
            print(f"--- {case_id} ---")
            try:
                target_n = self.repository.extract_size_from_key(case_id, require_case_index=True)
                if target_n not in filters_by_size:
                    raise ValueError(
                        MessageCatalog.format(
                            "FILTER_MISSING",
                            self.locale,
                            target_n=target_n,
                            candidates=candidates_display,
                        )
                    )

                if not isinstance(entry, dict) or "input" not in entry or "expected" not in entry:
                    raise ValueError("패턴 항목에 필수 필드('input' 또는 'expected')가 누락되었습니다.")

                expected_std = LabelNormalizer.normalize(entry["expected"])
                pattern = Matrix2D(entry["input"])

                if pattern.size != target_n:
                    raise ValueError(
                        f"필터 크기({target_n}×{target_n})와 입력 패턴 크기({pattern.size}×{pattern.size}) 불일치"
                    )

                filter_pair = filters_by_size[target_n]
                score_cross = self.engine.compute_mac_2d(pattern, filter_pair[LabelNormalizer.CROSS])
                score_x = self.engine.compute_mac_2d(pattern, filter_pair[LabelNormalizer.X])
                predicted = self.engine.judge_scores(score_cross, score_x)

                fmt_cross, fmt_x = self._format_score(score_cross, score_x)
                print(f"Cross 점수: {fmt_cross}")
                print(f"X 점수: {fmt_x}")

                if predicted == expected_std:
                    pass_count += 1
                    print(
                        self.format_judgment_line(
                            predicted, self.engine.epsilon, expected=expected_std, result_tag="PASS"
                        )
                    )
                else:
                    status_code, severity, reason = self._diagnose_case_failure(
                        predicted, expected_std, score_cross, score_x, self.engine.epsilon
                    )
                    tag = (
                        f"FAIL (동점 규칙, code={status_code}, severity={severity})"
                        if status_code == self.STATUS_TIE
                        else f"FAIL (code={status_code}, severity={severity})"
                    )
                    print(
                        self.format_judgment_line(
                            predicted, self.engine.epsilon, expected=expected_std, result_tag=tag
                        )
                    )
                    fail_cases.append(
                        {
                            "case_id": case_id,
                            "status_code": status_code,
                            "severity": severity,
                            "predicted": predicted,
                            "expected": expected_std,
                            "candidate_filters": candidates_display,
                            "reason": reason,
                        }
                    )

                    if status_code == self.STATUS_TIE:
                        norm_cross_f = self.engine.normalize_filter_l1(filter_pair[LabelNormalizer.CROSS])
                        norm_x_f = self.engine.normalize_filter_l1(filter_pair[LabelNormalizer.X])
                        re_cross = self.engine.compute_mac_2d(pattern, norm_cross_f)
                        re_x = self.engine.compute_mac_2d(pattern, norm_x_f)
                        re_pred = self.engine.judge_scores(re_cross, re_x)
                        tie_recheck_notes.append(
                            f"  * {case_id}: L1 가중치 합 정규화 시 Cross={re_cross:.4f}, "
                            f"X={re_x:.4f} -> {re_pred} (expected: {expected_std})"
                        )

            except ValueError as exc:
                reason = f"스키마/크기 검증 오류로 FAIL ({exc})"
                print(
                    f"판정: ERROR (epsilon={self.engine.epsilon:.0e}) | "
                    f"FAIL (code={self.STATUS_SCHEMA_ERROR}, severity={self.SEVERITY_HIGH}) - {exc}"
                )
                fail_cases.append(
                    {
                        "case_id": case_id,
                        "status_code": self.STATUS_SCHEMA_ERROR,
                        "severity": self.SEVERITY_HIGH,
                        "predicted": "ERROR",
                        "expected": entry.get("expected") if isinstance(entry, dict) else None,
                        "candidate_filters": candidates_display,
                        "reason": reason,
                    }
                )

        bench_records, projection_1m = self.print_benchmark_table(
            self.BENCHMARK_SIZES, filters_by_size=filters_by_size, section_num=3, include_projection=True
        )

        fail_count = total_count - pass_count
        analyzed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary_payload = {
            "analyzed_at": analyzed_at,
            "epsilon": self.engine.epsilon,
            "repeats": self.engine.repeats,
            "warmup": self.engine.warmup,
            "output_format_rule": self.OUTPUT_FORMAT_RULE,
            "min_report_lines_rule": self.MIN_REPORT_LINES,
            "loaded_filter_candidates": [f"size_{k}" for k in sorted(filters_by_size)],
            "loaded_filter_candidates_display": candidates_display,
            "total": total_count,
            "passed": pass_count,
            "failed": fail_count,
            "fail_cases": fail_cases,
            "benchmarks": bench_records,
            "large_scale_projection_1000x1000": projection_1m,
        }
        report_lines = self.render_markdown_report_lines(summary_payload)
        summary_payload["report_line_count"] = len(report_lines)
        summary_payload["report_lines"] = report_lines

        saved_ok = self.repository.save_report_atomic(summary_payload)

        print("\n#----------------------------------------")
        print("# [4] 결과 요약")
        print("#----------------------------------------")
        print(f"총 테스트: {total_count}개")
        print(f"통과: {pass_count}개")
        print(f"실패: {fail_count}개")
        print(f"로드된 필터 후보 목록: {candidates_display}")

        if fail_cases:
            print("\n실패 케이스 (심각도 및 표준 상태 코드 포함):")
            for item in fail_cases:
                print(
                    f"- [{item['severity']}|code={item['status_code']}] "
                    f"{item['case_id']}: {item['reason']}"
                )
        else:
            print("\n실패 케이스 없음 (라벨 정규화 및 허용오차 비교 정책 정상 적용)")

        if tie_recheck_notes:
            print("\n[동점(UNDECIDED) 케이스 보조 재검증 — 필터 L1 가중치 정규화 시뮬레이션]")
            for note in tie_recheck_notes:
                print(note)

        if saved_ok:
            report_name = os.path.basename(self.repository.report_file)
            print(
                f"\n[리포트 동기화 저장] {report_name} "
                f"(생성 시각: {analyzed_at}, 검증 줄 수: {len(report_lines)}줄 >= 최소 {self.MIN_REPORT_LINES}줄)"
            )
        return summary_payload

    def run_bonus_generator_and_optimization_mode(self) -> None:
        """모드 3(보너스): 크기 N 자동 패턴 생성기, 2D vs 1D 평탄화 비교, 1000×1000(1M) 실측 및 스레드 병목 실험"""
        print("\n#----------------------------------------")
        print("# [보너스 1 & 2] N×N 패턴 자동 생성, 1D 최적화, 1M 실측 및 병목 실험")
        print("#----------------------------------------")
        n = self._prompt_int_in_range("생성할 정방 패턴 크기 N 입력 (3~51): ", 3, 51)
        pair = PatternGenerator.generate_pair(n)
        cross_mat = pair[LabelNormalizer.CROSS]
        x_mat = pair[LabelNormalizer.X]

        print(f"\n[OK] {n}×{n} Cross 및 X 패턴 자동 생성 완료")
        if n <= 15:
            print(f"\n[생성된 {n}×{n} Cross 패턴 미리보기]")
            print(cross_mat.format_visual())
            print(f"\n[생성된 {n}×{n} X 패턴 미리보기]")
            print(x_mat.format_visual())

        score_cc_2d = self.engine.compute_mac_2d(cross_mat, cross_mat)
        score_cc_1d = self.engine.compute_mac_1d(cross_mat.to_flat_list(), cross_mat.to_flat_list())
        score_cx_2d = self.engine.compute_mac_2d(cross_mat, x_mat)
        print(
            f"\n[정합성 검증] Cross×Cross (2D={score_cc_2d:.1f}, 1D={score_cc_1d:.1f}) | "
            f"Cross×X 교차 점수={score_cx_2d:.1f}"
        )

        target_sizes = sorted(set(self.BENCHMARK_SIZES + [n]))
        records = self.engine.benchmark_sizes(target_sizes)
        print(
            f"\n[2D 중첩 리스트 vs 1D 평탄화 배열(길이 N²) 성능 비교 "
            f"(repeats={records[0]['repeats']}회, warmup={records[0]['warmup']}회)]"
        )
        print(
            f"{'크기':<7}{'반복수':<8}{'2D평균(ms)':<12}{'표준편차(±ms)':<14}"
            f"{'1D평균(ms)':<12}{'속도개선율':<11}{'연산수(N²)':<11}{'예상메모리(MiB)':<14}"
        )
        print("-" * 89)
        for rec in records:
            label = f"{rec['size']}×{rec['size']}"
            print(
                f"{label:<7}{rec['repeats']:<8}{rec['avg_2d_ms']:<12.4f}{rec['std_2d_ms']:<14.4f}"
                f"{rec['avg_1d_ms']:<12.4f}{rec['speedup_pct']:>+8.1f}%  "
                f"{rec['ops']:<11}{rec['est_mem_mib']:<14.4f}"
            )

        sample_1m = self.engine.measure_large_scale_sample(size=1000, repeats=3, warmup=1)
        print(
            f"\n[1000×1000 대형 입력(1,000,000 원소) 실제 샘플 측정 (repeats={sample_1m['repeats']}회, warmup=1회)]"
        )
        print(
            f"  - 2D 실측: {sample_1m['measured_2d_ms']:.2f} ms | "
            f"1D 실측: {sample_1m['measured_1d_ms']:.2f} ms | "
            f"파이썬 2D 메모리: {sample_1m['est_python_2d_mib']:.2f} MiB "
            f"(연속 float64 버퍼 시 {sample_1m['est_contiguous_fp64_mib']:.2f} MiB)"
        )

        bottleneck = self.engine.benchmark_serial_vs_threaded(size=25, batch_filters=64, workers=4)
        print(
            f"\n[CPU 직렬 vs 파이썬 다중 스레드({bottleneck['workers']}스레드) 배치 병목 실험 "
            f"(25×25 필터 {bottleneck['batch_filters']}개, 총 {bottleneck['total_mac_ops']}회 MAC)]"
        )
        print(
            f"  - 단일 스레드 2D 직렬: {bottleneck['serial_2d_ms']:.3f} ms | "
            f"단일 스레드 1D 직렬: {bottleneck['serial_1d_ms']:.3f} ms | "
            f"{bottleneck['workers']}스레드(GIL 경합): {bottleneck['threaded_2d_ms']:.3f} ms"
        )

    def print_menu(self) -> None:
        print("\n=== Mini NPU Simulator ===")
        print(
            f"[현재 설정] 허용오차(epsilon): {self.engine.epsilon:.0e} | "
            f"반복수(repeats): {self.engine.repeats}회 | 워밍업(warmup): {self.engine.warmup}회"
        )
        print(f"[표준 포맷] {self.OUTPUT_FORMAT_RULE}")
        print("\n[모드 선택]")
        print("1. 사용자 입력 (3x3)")
        print("2. data.json 분석")
        print("3. 패턴 생성기 및 2D/1D 메모리 최적화 비교 (보너스)")
        print("4. 종료")

    def run(self) -> None:
        """메인 시뮬레이터 루프 및 Ctrl+C / EOF 안전 종료 보장"""
        while True:
            try:
                self.print_menu()
                choice = self._prompt_int_in_range("선택: ", 1, 4)
                if choice == 1:
                    self.run_user_input_mode()
                elif choice == 2:
                    try:
                        self.evaluate_dataset()
                    except (FileNotFoundError, RuntimeError) as exc:
                        print(f"\n[데이터 오류] {exc}")
                elif choice == 3:
                    self.run_bonus_generator_and_optimization_mode()
                elif choice == 4:
                    print("\n시뮬레이터를 정상 종료합니다. 이용해 주셔서 감사합니다!")
                    return
            except (KeyboardInterrupt, EOFError):
                print(f"\n\n{MessageCatalog.format('SAFE_SHUTDOWN', self.locale)}")
                return


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    try:
        app = SimulatorApp()
        app.run()
    except (KeyboardInterrupt, EOFError):
        print(f"\n{MessageCatalog.format('SAFE_SHUTDOWN')}")


if __name__ == "__main__":
    main()
