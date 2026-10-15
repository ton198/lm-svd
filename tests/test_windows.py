import unittest

import numpy as np

from data_pipline.windows import select_windows


class SelectWindowsTests(unittest.TestCase):
    def test_windows_are_exact_grouped_and_non_overlapping(self) -> None:
        stream = np.arange(200)
        windows = select_windows(stream, 3, 8, 2, 4, 7)

        self.assertEqual(len(windows), 5)
        self.assertEqual([window.group for window in windows], ["history"] * 3 + ["future"] * 2)
        for window in windows:
            expected_length = 8 if window.group == "history" else 4
            self.assertEqual(len(window.token_ids), expected_length)
            np.testing.assert_array_equal(
                window.token_ids,
                stream[window.start_token : window.start_token + expected_length],
            )
        for group in ("history", "future"):
            starts = [window.start_token for window in windows if window.group == group]
            self.assertEqual(starts, sorted(starts))
        ordered = sorted(windows, key=lambda window: window.start_token)
        for current, following in zip(ordered, ordered[1:]):
            self.assertLessEqual(
                current.start_token + len(current.token_ids), following.start_token
            )

    def test_same_seed_produces_identical_windows(self) -> None:
        stream = np.arange(200)
        first = select_windows(stream, 3, 8, 2, 4, 7)
        second = select_windows(stream, 3, 8, 2, 4, 7)
        self.assertEqual(
            [(window.group, window.start_token, window.token_ids.tolist()) for window in first],
            [(window.group, window.start_token, window.token_ids.tolist()) for window in second],
        )

    def test_insufficient_slots_report_capacity_error(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "not enough distinct non-overlapping windows"
        ):
            select_windows(np.arange(16), 3, 8, 0, 4, 0)

    def test_duplicate_sequences_are_skipped_before_capacity_error(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "not enough distinct non-overlapping windows"
        ) as raised:
            select_windows(np.zeros(32, dtype=np.uint16), 2, 8, 0, 4, 0)
        self.assertIn("selected 1", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
