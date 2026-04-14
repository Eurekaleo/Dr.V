from __future__ import annotations

import unittest

from drv_agent.runner_utils import build_dense_caption_prompt, parse_claim_payload, select_evenly_spaced_indices


class RunnerUtilsTest(unittest.TestCase):
    def test_parse_claim_payload_accepts_plain_text(self) -> None:
        self.assertEqual(parse_claim_payload("the child picks up the book"), {"claim": "the child picks up the book"})

    def test_parse_claim_payload_accepts_json(self) -> None:
        self.assertEqual(
            parse_claim_payload('{"claim": "the child picks up the book", "score": 0.5}'),
            {"claim": "the child picks up the book", "score": 0.5},
        )

    def test_select_evenly_spaced_indices_deduplicates_short_ranges(self) -> None:
        self.assertEqual(select_evenly_spaced_indices(0, 2, 8), [0, 1, 2])

    def test_select_evenly_spaced_indices_spans_full_range(self) -> None:
        self.assertEqual(select_evenly_spaced_indices(10, 20, 4), [10, 13, 17, 20])

    def test_build_dense_caption_prompt_embeds_claim_and_event(self) -> None:
        prompt = build_dense_caption_prompt(
            claim="the child picks up the book",
            event_name="child reaches toward the shelf",
            timestamps=[1.0, 2.5],
        )
        self.assertIn("Claim to verify: the child picks up the book.", prompt)
        self.assertIn("Target event: child reaches toward the shelf.", prompt)
        self.assertIn("1.00s, 2.50s", prompt)


if __name__ == "__main__":
    unittest.main()
