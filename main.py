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
    """외부 라이브러리 없이 순수 반복문 기반 MAC(Multiply-Accumulate) 연산을 수행하는 코어 엔진 (SRP)"""

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


