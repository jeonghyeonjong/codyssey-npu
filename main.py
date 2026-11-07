from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import math
import os
import shutil
import sys
import time


class Matrix2D:
    """N×N 2차원 정방 숫자 배열(패턴·필터)을 표현하고 셀 단위 접근 및 차원 무결성을 보장하는 도메인 클래스 (SRP)"""

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
        """메모리 접근 단순화(보너스 최적화)를 위해 길이 N²의 1차원 연속 리스트로 변환"""
        return [val for row in self._grid for val in row]

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

    _LABEL_MAP = {
        "+": CROSS,
        "cross": CROSS,
        "x": X,
    }

    @classmethod
    def allowed_aliases_summary(cls) -> str:
        """정규화 실패 시 사용자에게 안내할 표준 라벨별 허용 입력 표기 요약"""
        return "Cross 계열: '+', 'cross', 'CROSS' / X 계열: 'x', 'X'"

    @classmethod
    def normalize(cls, raw_label: str) -> str:
        """'+'/'cross' -> 'Cross', 'x' -> 'X' 표준 라벨 변환 (대소문자·앞뒤 공백 무시)"""
        if not isinstance(raw_label, str) or not raw_label.strip():
            raise ValueError(
                f"라벨 값이 비어 있거나 문자열이 아닙니다. (허용 표기 — {cls.allowed_aliases_summary()})"
            )
        key = raw_label.strip().lower()
        if key not in cls._LABEL_MAP:
            raise ValueError(
                f"알 수 없는 비표준 라벨입니다: {raw_label!r} (허용 표기 — {cls.allowed_aliases_summary()})"
            )
        return cls._LABEL_MAP[key]


class MacEngine:
    """순수 반복문 기반 2D/1D MAC 연산, 허용오차(epsilon) 동점 판정 및 연산 시간 측정을 전담하는 엔진 (SRP)"""

    DEFAULT_EPSILON = 1e-9
    DEFAULT_REPEATS = 1000

    def __init__(self, epsilon: float = DEFAULT_EPSILON, repeats: int = DEFAULT_REPEATS):
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or epsilon <= 0:
            raise ValueError("허용오차(epsilon)는 0보다 큰 양수여야 합니다.")
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 10:
            raise ValueError("반복 측정 횟수(repeats)는 최소 10 이상의 정수여야 합니다.")
        self.epsilon = float(epsilon)
        self.repeats = repeats

    @staticmethod
    def compute_mac_2d(pattern: Matrix2D, filter_mat: Matrix2D) -> float:
        """2차원 중첩 반복문으로 같은 위치 원소끼리 곱해 순차 누적 합산(MAC) 점수 반환"""
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
        """1차원 평탄화 배열(길이 N²)에 대해 단일 반복문으로 MAC 연산 수행 (메모리 접근 최적화)"""
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

    def measure_pair_mac_ms(
        self,
        pattern: Matrix2D,
        filter_a: Matrix2D,
        filter_b: Matrix2D,
        repeats: int | None = None,
    ) -> tuple[float, float, float]:
        """콘솔/파일 I/O를 배제하고 순수 MAC 연산 호출 구간의 평균 소요 시간(ms)과 점수를 측정"""
        iters = max(10, repeats if repeats is not None else self.repeats)
        score_a = 0.0
        score_b = 0.0
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
    ) -> list[dict]:
        """크기별(3×3 ~ 25×25 등) 단일 MAC 연산의 2D 중첩 반복 vs 1D 평탄화 평균 시간(ms) 및 연산 횟수(N²) 측정"""
        iters = max(10, repeats if repeats is not None else self.repeats)
        results: list[dict] = []

        for size in sizes:
            pattern = PatternGenerator.generate_cross(size)
            if filters_by_size and size in filters_by_size and LabelNormalizer.CROSS in filters_by_size[size]:
                filter_mat = filters_by_size[size][LabelNormalizer.CROSS]
            else:
                filter_mat = PatternGenerator.generate_cross(size)

            # I/O 및 to_flat_list() 변환 비용을 측정 구간에서 명확히 제외하기 위해 사전 변환
            flat_p = pattern.to_flat_list()
            flat_f = filter_mat.to_flat_list()

            t0 = time.perf_counter()
            for _ in range(iters):
                self.compute_mac_2d(pattern, filter_mat)
            avg_2d_ms = ((time.perf_counter() - t0) / iters) * 1000.0

            t1 = time.perf_counter()
            for _ in range(iters):
                self.compute_mac_1d(flat_p, flat_f)
            avg_1d_ms = ((time.perf_counter() - t1) / iters) * 1000.0

            speedup_pct = ((avg_2d_ms - avg_1d_ms) / avg_2d_ms * 100.0) if avg_2d_ms > 0 else 0.0
            results.append(
                {
                    "size": size,
                    "ops": size * size,
                    "repeats": iters,
                    "avg_2d_ms": avg_2d_ms,
                    "avg_1d_ms": avg_1d_ms,
                    "speedup_pct": speedup_pct,
                }
            )
        return results

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
    """data.json 로드·키 규칙 해석·백업 복구(.bak) 및 실행 리포트 원자적 저장(os.replace)을 전담하는 클래스 (SRP)"""

    KEY_RULE_GUIDE = "권장 예시: 필터 키 'size_5', 패턴 키 'size_5_1', 'size_13_2'"

    def __init__(self, base_dir: str | None = None, data_filename: str = "data.json"):
        root = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.data_file = os.path.join(root, data_filename)
        self.data_backup_file = f"{self.data_file}.bak"
        self.data_corrupt_file = f"{self.data_file}.corrupt"
        self.report_file = os.path.join(root, "report_state.json")
        self.report_backup_file = os.path.join(root, "report_state.json.bak")
        self.report_temp_file = os.path.join(root, "report_state.json.tmp")

    @classmethod
    def extract_size_from_key(cls, key: str, require_case_index: bool = False) -> int:
        """'size_5'(필터 키) 또는 'size_13_2'(패턴 키)에서 행렬 크기 정수 N을 추출하고 위반 사유별 안내 반환"""
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"키가 비어 있거나 문자열이 아닙니다 ({cls.KEY_RULE_GUIDE}).")
        if not key.startswith("size_"):
            raise ValueError(
                f"키 접두사('size_') 누락: {key!r} ({cls.KEY_RULE_GUIDE})."
            )
        parts = key.split("_")
        if len(parts) < 2 or not parts[1]:
            raise ValueError(
                f"키에 크기 숫자 N 구분자가 없습니다: {key!r} ({cls.KEY_RULE_GUIDE})."
            )
        if not parts[1].lstrip("-").isdigit():
            raise ValueError(
                f"크기 N 위치에 정수가 아닌 문자열({parts[1]!r})이 지정되었습니다: {key!r} ({cls.KEY_RULE_GUIDE})."
            )
        size = int(parts[1])
        if size <= 0:
            raise ValueError(
                f"크기 N({size})은 1 이상의 양의 정수여야 합니다: {key!r} ({cls.KEY_RULE_GUIDE})."
            )
        if require_case_index and (len(parts) < 3 or not parts[2]):
            raise ValueError(
                f"패턴 식별자 일련번호('_{{idx}}')가 누락되었습니다: {key!r} ({cls.KEY_RULE_GUIDE})."
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

                for req_label in (LabelNormalizer.CROSS, LabelNormalizer.X):
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

    def __init__(
        self,
        repository: DatasetRepository | None = None,
        engine: MacEngine | None = None,
    ):
        self.repository = repository or DatasetRepository()
        self.engine = engine or MacEngine()

    @staticmethod
    def _format_score(score_a: float, score_b: float) -> tuple[str, str]:
        """두 점수가 근접한 동점 경계 케이스에서 IEEE-754 소수점 16자리 정밀도를 그대로 노출"""
        if abs(score_a - score_b) < 1e-6:
            return f"{score_a:.16f}", f"{score_b:.16f}"
        return f"{score_a:.4f}".rstrip("0").rstrip("."), f"{score_b:.4f}".rstrip("0").rstrip(".")

    @staticmethod
    def _diagnose_case_failure(
        predicted: str,
        expected: str,
        score_cross: float,
        score_x: float,
        epsilon: float,
    ) -> str:
        """판정 불일치 발생 시 수치 비교(동점 규칙) vs 반대 라벨 오분류 원인을 명확히 요약"""
        diff = abs(score_cross - score_x)
        if predicted == LabelNormalizer.UNDECIDED:
            return (
                f"동점(UNDECIDED) 처리 규칙에 따라 FAIL "
                f"(|Cross-X| = {diff:.2e} < {epsilon:.0e}, expected: {expected})"
            )
        return (
            f"예측 라벨 불일치로 FAIL "
            f"(판정: {predicted}, expected: {expected}, Cross={score_cross:.4f}, X={score_x:.4f})"
        )

    @staticmethod
    def _prompt_int_in_range(prompt: str, min_val: int, max_val: int) -> int:
        while True:
            raw = input(prompt).strip()
            if not raw:
                print(f"[입력 오류] 입력이 비어 있습니다. {min_val}~{max_val} 사이의 숫자를 입력하세요.")
                continue
            try:
                value = int(raw)
            except ValueError:
                print(f"[입력 오류] 숫자가 아닙니다. {min_val}~{max_val} 사이의 정수를 입력하세요.")
                continue
            if min_val <= value <= max_val:
                return value
            print(f"[입력 오류] 허용 범위를 벗어났습니다. {min_val}~{max_val} 사이의 숫자를 입력하세요.")

    @staticmethod
    def read_matrix_from_console(title: str, size: int = 3) -> Matrix2D:
        """지정된 크기(기본 3×3)의 행렬을 한 줄씩 입력받아 열 개수 불일치·비숫자 원인을 구분 안내 후 재입력 유도"""
        print(f"\n{title} ({size}줄 입력, 공백 구분)")
        rows: list[list[float]] = []
        for row_idx in range(1, size + 1):
            while True:
                line = input(f"  {row_idx}행: ").strip()
                if not line:
                    print(
                        f"입력 형식 오류 [빈 줄 입력]: 각 줄에 {size}개의 숫자를 공백으로 구분해 입력하세요."
                    )
                    continue
                tokens = line.split()
                if len(tokens) != size:
                    print(
                        f"입력 형식 오류 [열 개수 불일치: 입력 {len(tokens)}개 != 기대 {size}개]: "
                        f"각 줄에 {size}개의 숫자를 공백으로 구분해 입력하세요."
                    )
                    continue

                parsed_row: list[float] = []
                bad_reason: str | None = None
                for tok in tokens:
                    try:
                        num = float(tok)
                    except ValueError:
                        bad_reason = f"숫자 파싱 실패({tok!r})"
                        break
                    try:
                        parsed_row.append(Matrix2D._validate_number(num, f"{row_idx}행"))
                    except ValueError:
                        bad_reason = f"유한 실수 아님({tok!r})"
                        break

                if bad_reason is not None:
                    print(
                        f"입력 형식 오류 [{bad_reason}]: "
                        f"각 줄에 {size}개의 숫자를 공백으로 구분해 입력하세요."
                    )
                    continue

                rows.append(parsed_row)
                break
        return Matrix2D(rows)

    def print_benchmark_table(
        self,
        sizes: list[int],
        filters_by_size: dict[int, dict[str, Matrix2D]] | None = None,
        section_num: int = 3,
    ) -> list[dict]:
        """크기별 반복수(repeats), 평균 MAC 연산 시간(2D vs 1D 최적화, ms) 및 연산 횟수(N²) 표 출력"""
        records = self.engine.benchmark_sizes(sizes, filters_by_size=filters_by_size)
        repeats = records[0]["repeats"] if records else self.engine.repeats

        print("\n#----------------------------------------")
        print(f"# [{section_num}] 성능 분석 (반복 측정 repeats={repeats}회 평균)")
        print("#----------------------------------------")
        print(
            f"{'크기':<8}{'반복수(회)':<12}{'평균 시간(ms)':<16}"
            f"{'1D 최적화(ms)':<16}{'연산 횟수(N²)':<14}"
        )
        print("-" * 66)
        for rec in records:
            label = f"{rec['size']}×{rec['size']}"
            print(
                f"{label:<8}{rec['repeats']:<12}{rec['avg_2d_ms']:<16.4f}"
                f"{rec['avg_1d_ms']:<16.4f}{rec['ops']:<14}"
            )
        return records

    def run_user_input_mode(self) -> None:
        """모드 1: 3×3 필터 A, B 입력 -> 저장 확인 -> 패턴 입력 -> MAC 연산/판정 -> 성능 분석(3×3) 출력"""
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

        print("\n#----------------------------------------")
        print("# [3] MAC 결과")
        print("#----------------------------------------")
        print(f"A 점수: {fmt_a}")
        print(f"B 점수: {fmt_b}")
        print(f"연산 시간(평균/{self.engine.repeats}회): {avg_pair_ms:.4f} ms")
        if judgment == "판정 불가":
            print(f"판정: 판정 불가 (|A-B| < {self.engine.epsilon:.0e})")
        else:
            print(f"판정: {judgment}")

        self.print_benchmark_table([3], section_num=4)

    def evaluate_dataset(self) -> dict:
        """모드 2: data.json 로드 -> 라벨 정규화 및 크기 검증 -> 케이스별 MAC 판정(케이스 단위 FAIL 격리) -> 성능·요약 출력"""
        raw_data = self.repository.load_dataset()

        print("\n#----------------------------------------")
        print("# [1] 필터 로드")
        print("#----------------------------------------")
        filters_by_size, filter_logs = self.repository.load_normalized_filters(raw_data["filters"])
        for msg in filter_logs:
            print(msg)

        available_sizes_str = (
            ", ".join(f"size_{k}" for k in sorted(filters_by_size)) if filters_by_size else "없음"
        )

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
                        f"매칭되는 'size_{target_n}' 필터가 없습니다 "
                        f"(현재 로드된 필터 후보: [{available_sizes_str}])."
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
                    print(f"판정: {predicted} | expected: {expected_std} | PASS")
                else:
                    reason = self._diagnose_case_failure(
                        predicted, expected_std, score_cross, score_x, self.engine.epsilon
                    )
                    tag = "FAIL (동점 규칙)" if predicted == LabelNormalizer.UNDECIDED else "FAIL"
                    print(f"판정: {predicted} | expected: {expected_std} | {tag}")
                    fail_cases.append({"case_id": case_id, "reason": reason})

                    if predicted == LabelNormalizer.UNDECIDED:
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
                print(f"판정: ERROR | FAIL - {exc}")
                fail_cases.append({"case_id": case_id, "reason": reason})

        bench_records = self.print_benchmark_table(
            self.BENCHMARK_SIZES, filters_by_size=filters_by_size, section_num=3
        )

        fail_count = total_count - pass_count
        analyzed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary_payload = {
            "analyzed_at": analyzed_at,
            "epsilon": self.engine.epsilon,
            "repeats": self.engine.repeats,
            "total": total_count,
            "passed": pass_count,
            "failed": fail_count,
            "fail_cases": fail_cases,
            "benchmarks": bench_records,
        }
        saved_ok = self.repository.save_report_atomic(summary_payload)

        print("\n#----------------------------------------")
        print("# [4] 결과 요약")
        print("#----------------------------------------")
        print(f"총 테스트: {total_count}개")
        print(f"통과: {pass_count}개")
        print(f"실패: {fail_count}개")

        if fail_cases:
            print("\n실패 케이스:")
            for item in fail_cases:
                print(f"- {item['case_id']}: {item['reason']}")
        else:
            print("\n실패 케이스 없음 (라벨 정규화 및 허용오차 비교 정책 정상 적용)")

        if tie_recheck_notes:
            print("\n[동점(UNDECIDED) 케이스 보조 재검증 — 필터 L1 가중치 정규화 시뮬레이션]")
            for note in tie_recheck_notes:
                print(note)

        if saved_ok:
            report_name = os.path.basename(self.repository.report_file)
            print(f"\n[리포트 동기화 저장] {report_name} (생성 시각: {analyzed_at})")
        return summary_payload

    def run_bonus_generator_and_optimization_mode(self) -> None:
        """모드 3(보너스): 크기 N 자동 패턴 생성기, 2D vs 1D 평탄화 비교 및 CPU 직렬 vs 스레드 병목 실험"""
        print("\n#----------------------------------------")
        print("# [보너스 1 & 2] N×N 패턴 자동 생성, 1D 최적화 및 병목 실험")
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
        print(f"\n[2D 중첩 리스트 vs 1D 평탄화 배열(길이 N²) 성능 비교 (repeats={records[0]['repeats']}회 평균)]")
        print(
            f"{'크기':<8}{'반복수(회)':<11}{'2D 평균(ms)':<14}"
            f"{'1D 평균(ms)':<14}{'속도 개선율':<12}{'연산 횟수(N²)':<12}"
        )
        print("-" * 71)
        for rec in records:
            label = f"{rec['size']}×{rec['size']}"
            print(
                f"{label:<8}{rec['repeats']:<11}{rec['avg_2d_ms']:<14.4f}{rec['avg_1d_ms']:<14.4f}"
                f"{rec['speedup_pct']:>+8.1f}%   {rec['ops']:<12}"
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
            f"성능 측정 반복수(repeats): {self.engine.repeats}회"
        )
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
                print("\n\n[안전 종료] 종료 신호가 감지되어 시뮬레이터를 안전하게 종료합니다.")
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
        print("\n[안전 종료] 프로그램을 정상 종료합니다.")


if __name__ == "__main__":
    main()
