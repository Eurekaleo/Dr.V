from __future__ import annotations

import unittest

from drv_agent.config import AgentConfig, RunSettings, build_agent
from drv_agent.schemas import DrVRequest, QAInput


class MockPipelineTest(unittest.TestCase):
    def test_mock_pipeline_returns_structured_report(self) -> None:
        agent = build_agent(AgentConfig(run=RunSettings(mode="mock")))
        request = DrVRequest(
            video_path="unused.mp4",
            qa=QAInput(
                question="Why did the baby walk towards the bookshelf?",
                options=[
                    "A. to pick up a music player",
                    "B. to play with man",
                    "C. to find a toy",
                    "D. to pick up a book",
                ],
            ),
            lvm_answer="B. to play with man",
        )
        report = agent.run(request)

        self.assertEqual(report.classification.level.value, "cognitive")
        self.assertTrue(report.assessment.has_hallucination)
        self.assertIn("recommendations", report.to_dict()["feedback"])


if __name__ == "__main__":
    unittest.main()
