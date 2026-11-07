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
    def normalize(cls, raw_label: str) -> str:
        """'+'/'cross' -> 'Cross', 'x' -> 'X' 표준 라벨 변환 (대소문자·공백 무시)"""
        if not isinstance(raw_label, str) or not raw_label.strip():
            raise ValueError("라벨 값이 비어 있거나 문자열이 아닙니다.")
        key = raw_label.strip().lower()
        if key not in cls._LABEL_MAP:
            raise ValueError(f"알 수 없는 비표준 라벨입니다: {raw_label!r} (허용: '+', 'cross', 'x')")
        return cls._LABEL_MAP[key]


class MacEngine:
    """순수 반복문 기반 2D/1D MAC 연산, 허용오차(epsilon) 동점 판정 및 연산 시간 측정을 전담하는 엔진 (SRP)"""

    DEFAULT_EPSILON = 1e-9
    DEFAULT_REPEATS = 1000

    def __init__(self, epsilon: float = DEFAULT_EPSILON):
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or epsilon <= 0:
            raise ValueError("허용오차(epsilon)는 0보다 큰 양수여야 합니다.")
        self.epsilon = float(epsilon)

    @staticmethod
    def compute_mac_2d(pattern: Matrix2D, filter_mat: Matrix2D) -> float:
        """2차원 중첩 반복문으로 같은 위치 원소끼리 곱해 누적 합산(MAC) 점수 반환"""
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
        repeats: int = DEFAULT_REPEATS,
    ) -> tuple[float, float, float]:
        """콘솔/파일 I/O를 배제하고 순수 MAC 연산 호출 구간의 평균 소요 시간(ms)과 점수를 측정"""
        iters = max(10, repeats)
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
        repeats: int = DEFAULT_REPEATS,
    ) -> list[dict]:
        """크기별(3×3 ~ 25×25 등) 단일 MAC 연산의 2D 중첩 반복 vs 1D 평탄화 평균 시간(ms) 및 연산 횟수(N²) 측정"""
        iters = max(10, repeats)
        results: list[dict] = []

        for size in sizes:
            pattern = PatternGenerator.generate_cross(size)
            if filters_by_size and size in filters_by_size and LabelNormalizer.CROSS in filters_by_size[size]:
                filter_mat = filters_by_size[size][LabelNormalizer.CROSS]
            else:
                filter_mat = PatternGenerator.generate_cross(size)

            # I/O 및 변환 오버헤드를 측정 구간에서 제외하기 위해 사전에 1D 배열 준비
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

    def __init__(self, base_dir: str | None = None, data_filename: str = "data.json"):
        root = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.data_file = os.path.join(root, data_filename)
        self.data_backup_file = f"{self.data_file}.bak"
        self.data_corrupt_file = f"{self.data_file}.corrupt"
        self.report_file = os.path.join(root, "report_state.json")
        self.report_backup_file = os.path.join(root, "report_state.json.bak")
        self.report_temp_file = os.path.join(root, "report_state.json.tmp")

    @staticmethod
    def extract_size_from_key(key: str) -> int:
        """'size_5' 또는 'size_13_2' 형태의 키에서 행렬 크기 정수 N을 추출"""
        if not isinstance(key, str) or not key.startswith("size_"):
            raise ValueError(f"키 이름 규칙('size_{{N}}' 또는 'size_{{N}}_{{idx}}') 위반: {key!r}")
        parts = key.split("_")
        if len(parts) < 2 or not parts[1].isdigit():
            raise ValueError(f"키에서 유효한 크기 N을 추출할 수 없습니다: {key!r}")
        size = int(parts[1])
        if size <= 0:
            raise ValueError(f"크기 N은 양의 정수여야 합니다: {key!r}")
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
                expected_n = self.extract_size_from_key(size_key)
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


