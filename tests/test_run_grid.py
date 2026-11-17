import argparse
import unittest
from unittest.mock import ANY, call, patch

from scripts.eval.run_grid import config_basename, run_grid


class RunGridTests(unittest.TestCase):
    def test_config_basename_must_be_a_single_directory_name(self) -> None:
        self.assertEqual(config_basename("experiment"), "experiment")
        for value in ("", ".", "..", "runs/experiment", r"runs\experiment"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                config_basename(value)

    @patch("scripts.eval.run_grid.subprocess.run")
    def test_runs_all_sixteen_design_axis_combinations(self, run) -> None:
        run.return_value.returncode = 0

        self.assertEqual(run_grid("experiment"), 0)

        expected = []
        for precision in ("FP32", "FP16"):
            for n_threads in (1, 4):
                for rate in (1.0, 1.5, 2.0, 3.0):
                    config = f"experiment_{precision}_t{n_threads}_r{rate}"
                    expected.append(
                        call(
                            [
                                "uv",
                                "run",
                                ANY,
                                "--precision",
                                precision,
                                "--n-threads",
                                str(n_threads),
                                "--rate",
                                str(rate),
                                "--config",
                                config,
                            ],
                            cwd=ANY,
                            check=False,
                        )
                    )

        self.assertEqual(run.call_args_list, expected)

    @patch("scripts.eval.run_grid.subprocess.run")
    def test_stops_and_propagates_a_failed_configuration(self, run) -> None:
        run.return_value.returncode = 7

        self.assertEqual(run_grid("experiment"), 7)
        run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
