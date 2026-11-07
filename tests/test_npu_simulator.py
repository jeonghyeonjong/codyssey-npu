import io
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    DatasetRepository,
    LabelNormalizer,
    MacEngine,
    Matrix2D,
    PatternGenerator,
    SimulatorApp,
)


class TestMatrixAndNormalizer(unittest.TestCase):
    """Matrix2D 셀 접근·무결성 및 LabelNormalizer 표준화·별칭 안내 검증"""

    def test_matrix_get_set_and_validation(self):
        m = Matrix2D([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        self.assertEqual(m.size, 3)
        self.assertEqual(m.get_value(0, 1), 1.0)
        m.set_value(0, 0, 2.5)
        self.assertEqual(m.get_value(0, 0), 2.5)
        self.assertEqual(len(m.to_flat_list()), 9)

        with self.assertRaises(ValueError):
            Matrix2D([[1, 0], [1, 0, 1]])
        with self.assertRaises(ValueError):
            Matrix2D([[True, 0], [0, 1]])
        with self.assertRaises(ValueError):
            Matrix2D([[float("nan"), 0], [0, 1]])

    def test_label_normalizer_and_alias_guide(self):
        self.assertEqual(LabelNormalizer.normalize("+"), "Cross")
        self.assertEqual(LabelNormalizer.normalize("cross"), "Cross")
        self.assertEqual(LabelNormalizer.normalize(" CROSS "), "Cross")
        self.assertEqual(LabelNormalizer.normalize("x"), "X")
        self.assertEqual(LabelNormalizer.normalize("X"), "X")
        with self.assertRaises(ValueError) as ctx:
            LabelNormalizer.normalize("circle")
        self.assertIn("허용 표기", str(ctx.exception))


class TestMacEngineAndEpsilonPolicy(unittest.TestCase):
    """순수 반복문 2D/1D MAC 연산 일치성, L1 정규화 재검증 및 GIL 병목 실험 검증"""

    def test_mac_2d_and_1d_parity_on_3x3(self):
        engine = MacEngine(epsilon=1e-9, repeats=50)
        cross = PatternGenerator.generate_cross(3)
        x_pat = PatternGenerator.generate_x(3)

        self.assertAlmostEqual(engine.compute_mac_2d(cross, cross), 5.0)
        self.assertAlmostEqual(engine.compute_mac_2d(cross, x_pat), 1.0)
        self.assertAlmostEqual(
            engine.compute_mac_2d(cross, cross),
            engine.compute_mac_1d(cross.to_flat_list(), cross.to_flat_list()),
        )

    def test_floating_point_drift_tie_and_l1_recheck(self):
        engine = MacEngine(epsilon=1e-9)
        score_a = 0.9
        score_b = 0.0
        for _ in range(9):
            score_b += 0.1  # 0.8999999999999999 in sequential IEEE-754 accumulation
        self.assertNotEqual(score_a, score_b)
        self.assertEqual(engine.judge_scores(score_a, score_b), "UNDECIDED")

        # L1 가중치 합 정규화 적용 시 절댓값 총합이 1.0이 되는지 확인
        raw_flt = PatternGenerator.generate_cross(5)
        norm_flt = engine.normalize_filter_l1(raw_flt)
        self.assertAlmostEqual(sum(norm_flt.to_flat_list()), 1.0)

    def test_serial_vs_threaded_bottleneck_experiment(self):
        engine = MacEngine()
        res = engine.benchmark_serial_vs_threaded(size=13, batch_filters=16, workers=2)
        self.assertTrue(res["parity_ok"])
        self.assertEqual(res["total_mac_ops"], 13 * 13 * 16)


class TestDatasetEvaluationAndErrorIsolation(unittest.TestCase):
    """공식 data.json 평가, 비정상 키/후보 목록 안내, 케이스 단위 FAIL 격리 및 .bak 복구 검증"""

    def test_key_parser_error_variants(self):
        self.assertEqual(DatasetRepository.extract_size_from_key("size_5"), 5)
        self.assertEqual(
            DatasetRepository.extract_size_from_key("size_13_2", require_case_index=True), 13
        )
        for bad_key in ("", "pattern_5_1", "size_abc_1", "size_0_1", "size_5"):
            with self.assertRaises(ValueError) as ctx:
                DatasetRepository.extract_size_from_key(bad_key, require_case_index=True)
            self.assertIn("권장 예시", str(ctx.exception))

    def test_official_dataset_evaluation(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        app = SimulatorApp(repository=DatasetRepository(base_dir=repo_root))
        summary = app.evaluate_dataset()
        self.assertEqual(summary["total"], 6)
        self.assertEqual(summary["passed"], 3)
        self.assertEqual(summary["failed"], 3)
        failed_ids = [item["case_id"] for item in summary["fail_cases"]]
        self.assertEqual(failed_ids, ["size_5_1", "size_13_2", "size_25_1"])

    def test_case_level_fail_isolation_with_candidate_list_and_backup_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = {
                "meta": {"version": "1.0", "type": "json"},
                "filters": {
                    "size_5": {
                        "cross": PatternGenerator.generate_cross(5).to_list(),
                        "x": PatternGenerator.generate_x(5).to_list(),
                    }
                },
                "patterns": {
                    "size_5_1": {
                        "input": PatternGenerator.generate_cross(5).to_list(),
                        "expected": "+",
                    },
                    "size_5_2": {
                        "input": PatternGenerator.generate_cross(3).to_list(),
                        "expected": "+",
                    },
                    "size_9_1": {
                        "input": PatternGenerator.generate_cross(9).to_list(),
                        "expected": "+",
                    },
                },
            }
            data_path = os.path.join(tmp, "data.json")
            with open(data_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)

            repo = DatasetRepository(base_dir=tmp)
            app = SimulatorApp(repository=repo)
            summary = app.evaluate_dataset()
            self.assertEqual(summary["total"], 3)
            self.assertEqual(summary["passed"], 1)
            self.assertEqual(summary["failed"], 2)
            # 매칭 실패 로그에 현재 로드된 후보 목록([size_5]) 표기 검증
            self.assertIn("size_5", summary["fail_cases"][1]["reason"])
            self.assertTrue(os.path.exists(repo.report_file))

            # data.json 문법 파손 시 .corrupt 격리 및 .bak 자동 복원 확인
            with open(data_path, "w", encoding="utf-8") as f:
                f.write("{broken json")
            recovered = repo.load_dataset()
            self.assertIn("filters", recovered)
            self.assertTrue(os.path.exists(repo.data_corrupt_file))

    def test_mode1_specific_input_error_causes_and_graceful_eof(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = DatasetRepository(base_dir=tmp)
            app = SimulatorApp(repository=repo)
            inputs = [
                "",
                "abc",
                "1",
                "",             # 빈 줄 입력 오류 유도
                "0 1",          # 열 개수 불일치 오류 유도
                "0 a 0",        # 숫자 파싱 실패 오류 유도
                "0 1 0",
                "1 1 1",
                "0 1 0",
                "1 0 1",
                "0 1 0",
                "1 0 1",
                "1 0 1",
                "0 1 0",
                "1 0 1",
                EOFError(),
            ]
            buf = io.StringIO()
            with patch("sys.stdout", buf), patch("builtins.input", side_effect=inputs):
                app.run()
            output = buf.getvalue()
            self.assertIn("열 개수 불일치", output)
            self.assertIn("숫자 파싱 실패", output)


if __name__ == "__main__":
    unittest.main()
