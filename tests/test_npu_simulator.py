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
    MessageCatalog,
    PatternGenerator,
    SimulatorApp,
)


class TestMatrixAndNormalizer(unittest.TestCase):
    """Matrix2D 셀 접근·메모리 추정·무결성 및 LabelNormalizer 표준화·한 줄 별칭 안내 검증"""

    def test_matrix_get_set_memory_and_validation(self):
        m = Matrix2D([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        self.assertEqual(m.size, 3)
        self.assertEqual(m.get_value(0, 1), 1.0)
        m.set_value(0, 0, 2.5)
        self.assertEqual(m.get_value(0, 0), 2.5)
        self.assertEqual(len(m.to_flat_list()), 9)

        mem_1000 = Matrix2D.estimate_pair_memory_mib(1000)
        self.assertGreater(mem_1000["python_2d_mib"], mem_1000["contiguous_fp64_mib"])

        with self.assertRaises(ValueError):
            Matrix2D([[1, 0], [1, 0, 1]])
        with self.assertRaises(ValueError):
            Matrix2D([[True, 0], [0, 1]])
        with self.assertRaises(ValueError):
            Matrix2D([[float("nan"), 0], [0, 1]])

    def test_label_normalizer_and_concise_alias_summary(self):
        self.assertEqual(LabelNormalizer.normalize("+"), "Cross")
        self.assertEqual(LabelNormalizer.normalize("cross"), "Cross")
        self.assertEqual(LabelNormalizer.normalize(" CROSS "), "Cross")
        self.assertEqual(LabelNormalizer.normalize("x"), "X")
        self.assertEqual(LabelNormalizer.normalize("X"), "X")
        with self.assertRaises(ValueError) as ctx:
            LabelNormalizer.normalize("circle")
        self.assertIn("허용 별칭 요약", str(ctx.exception))


class TestMacEngineWarmupStatsAndProjections(unittest.TestCase):
    """행 우선 2D/1D MAC 일치성, 워밍업·표준편차 통계, 점수 정규화 비율 및 GIL 병목 검증"""

    def test_mac_2d_1d_parity_and_score_bounds(self):
        engine = MacEngine(epsilon=1e-9, repeats=40, warmup=5)
        cross = PatternGenerator.generate_cross(3)
        x_pat = PatternGenerator.generate_x(3)

        self.assertAlmostEqual(engine.compute_mac_2d(cross, cross), 5.0)
        self.assertAlmostEqual(engine.compute_mac_2d(cross, x_pat), 1.0)
        self.assertAlmostEqual(
            engine.compute_mac_2d(cross, cross),
            engine.compute_mac_1d(cross.to_flat_list(), cross.to_flat_list()),
        )

        bounds = engine.compute_score_bounds_and_ratio(cross, cross)
        self.assertAlmostEqual(bounds["max_possible"], 5.0)
        self.assertAlmostEqual(bounds["normalized_ratio"], 1.0)

    def test_floating_point_drift_tie_and_l1_recheck(self):
        engine = MacEngine(epsilon=1e-9)
        score_a = 0.9
        score_b = 0.0
        for _ in range(9):
            score_b += 0.1  # 순차 행 우선 누적 시 0.8999999999999999
        self.assertNotEqual(score_a, score_b)
        self.assertEqual(engine.judge_scores(score_a, score_b), "UNDECIDED")

        raw_flt = PatternGenerator.generate_cross(5)
        norm_flt = engine.normalize_filter_l1(raw_flt)
        self.assertAlmostEqual(sum(norm_flt.to_flat_list()), 1.0)

    def test_benchmark_warmup_stddev_and_1m_projection(self):
        engine = MacEngine(epsilon=1e-9, repeats=40, warmup=5)
        records = engine.benchmark_sizes([3, 5], warmup=5)
        self.assertEqual(records[0]["warmup"], 5)
        self.assertGreaterEqual(records[0]["max_2d_ms"], records[0]["min_2d_ms"])
        self.assertGreaterEqual(records[0]["std_2d_ms"], 0.0)
        self.assertGreater(records[0]["est_mem_mib"], 0.0)

        proj = engine.project_large_scale(records[-1], target_size=1000)
        self.assertEqual(proj["ops"], 1_000_000)
        self.assertGreater(proj["projected_2d_ms"], 0.0)

    def test_serial_vs_threaded_bottleneck_experiment(self):
        engine = MacEngine()
        res = engine.benchmark_serial_vs_threaded(size=13, batch_filters=16, workers=2)
        self.assertTrue(res["parity_ok"])
        self.assertEqual(res["total_mac_ops"], 13 * 13 * 16)


class TestDatasetEvaluationReportCodesAndLocalization(unittest.TestCase):
    """공식 data.json 상태코드(tie)/심각도, 후보 포맷 일관성, 최소 10줄 검증 및 다국어 템플릿 확인"""

    def test_key_parser_diverse_error_variants(self):
        self.assertEqual(DatasetRepository.extract_size_from_key("size_5"), 5)
        self.assertEqual(
            DatasetRepository.extract_size_from_key("size_13_2", require_case_index=True), 13
        )
        diverse_bad_keys = (
            "",
            " size_5_1 ",
            "Size_5_1",
            "size-5-1",
            "size.5.1",
            "pattern_5_1",
            "size_abc_1",
            "size_0_1",
            "size_5",
        )
        for bad_key in diverse_bad_keys:
            with self.assertRaises(ValueError) as ctx:
                DatasetRepository.extract_size_from_key(bad_key, require_case_index=True)
            self.assertIn("권장 예시", str(ctx.exception))

    def test_official_dataset_tie_status_code_severity_and_min_lines(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        app = SimulatorApp(repository=DatasetRepository(base_dir=repo_root))
        summary = app.evaluate_dataset()
        self.assertEqual(summary["total"], 6)
        self.assertEqual(summary["passed"], 3)
        self.assertEqual(summary["failed"], 3)
        self.assertGreaterEqual(summary["report_line_count"], SimulatorApp.MIN_REPORT_LINES)
        self.assertEqual(summary["loaded_filter_candidates_display"], "[size_5, size_13, size_25]")

        for fc in summary["fail_cases"]:
            self.assertEqual(fc["status_code"], SimulatorApp.STATUS_TIE)
            self.assertEqual(fc["severity"], SimulatorApp.SEVERITY_LOW)
            self.assertEqual(fc["candidate_filters"], "[size_5, size_13, size_25]")

    def test_case_level_fail_isolation_candidate_consistency_and_backup_recovery(self):
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
            # 콘솔 메시지와 리포트 필드가 동일한 후보 형식('[size_5]')을 사용하는지 검증
            self.assertIn("[size_5]", summary["fail_cases"][1]["reason"])
            self.assertEqual(summary["fail_cases"][1]["candidate_filters"], "[size_5]")
            self.assertEqual(summary["fail_cases"][1]["status_code"], SimulatorApp.STATUS_SCHEMA_ERROR)
            self.assertEqual(summary["fail_cases"][1]["severity"], SimulatorApp.SEVERITY_HIGH)

            with open(data_path, "w", encoding="utf-8") as f:
                f.write("{broken json")
            recovered = repo.load_dataset()
            self.assertIn("filters", recovered)
            self.assertTrue(os.path.exists(repo.data_corrupt_file))

    def test_mode1_error_catalog_ko_and_en_locale(self):
        self.assertIn("column mismatch", MessageCatalog.format("ROW_COL_MISMATCH", "en", actual=2, size=3))
        with tempfile.TemporaryDirectory() as tmp:
            repo = DatasetRepository(base_dir=tmp)
            app = SimulatorApp(repository=repo, locale="ko")
            inputs = [
                "",
                "abc",
                "1",
                "",             # 빈 줄 입력 오류
                "0 1",          # 열 개수 불일치 오류
                "0 a 0",        # 숫자 파싱 실패 오류
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
            self.assertIn("기준 epsilon=1e-09", output)


if __name__ == "__main__":
    unittest.main()
