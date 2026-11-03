import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import gtm_agent as scoring


class ScoreProspectTests(unittest.TestCase):
    def setUp(self):
        self.offering = {
            "description": "Analytics platform",
            "required_tech_stack": ["Snowflake", "Tableau", "dbt", "Kafka"],
            "min_annual_revenue": 50000000,
            "target_segment": "Enterprise",
        }
        self.profile = {
            "annual_revenue": 50000000,
            "tech_stack": ["Snowflake", "Tableau", "dbt"],
            "segment": "Enterprise",
        }
        self.llm = self.enterContext(patch.object(scoring, "_scoring_llm"))
        self.llm.invoke.return_value = scoring.ScoreJustification(justification="Fit explanation")
        self.enterContext(patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}))

    def score(self):
        return scoring.score_prospect.invoke({
            "prospect_profile": self.profile, "offering": self.offering,
        })

    def test_identical_inputs_keep_scores_despite_different_model_outputs(self):
        self.llm.invoke.side_effect = [
            SimpleNamespace(justification="First explanation", score=1, rubric_breakdown={}),
            SimpleNamespace(justification="Second explanation", score=99, rubric_breakdown={}),
        ]
        first, second = self.score(), self.score()
        self.assertNotEqual(first.pop("justification"), second.pop("justification"))
        self.assertEqual(first, second)
        self.assertEqual(first["rubric_breakdown"], {
            "revenue_fit": 100, "tech_stack_match": 75, "segment_fit": 100,
            "component_max": 100,
        })
        self.assertEqual(first["score"], 90)

    def test_revenue_fit_and_weighted_overall(self):
        for revenue, expected in [(50000000, 100), (100000000, 100),
                                  (25000000, 50), (0, 0), (None, 0)]:
            with self.subTest(revenue=revenue):
                self.profile["annual_revenue"] = revenue
                result = self.score()
                self.assertEqual(result["rubric_breakdown"]["revenue_fit"], expected)
                self.assertEqual(result["score"], round(0.4 * expected + 0.4 * 75 + 0.2 * 100, 1))
        del self.profile["annual_revenue"]
        self.assertEqual(self.score()["rubric_breakdown"]["revenue_fit"], 0)

    def test_segment_policy(self):
        for segment, expected in [(" enterprise ", 100), ("Mid-Market", 0), (None, 50)]:
            with self.subTest(segment=segment):
                self.profile["segment"] = segment
                self.assertEqual(self.score()["rubric_breakdown"]["segment_fit"], expected)
        del self.profile["segment"]
        self.assertEqual(self.score()["rubric_breakdown"]["segment_fit"], 50)
        del self.offering["target_segment"]
        self.assertEqual(self.score()["rubric_breakdown"]["segment_fit"], 100)

    def test_zero_threshold_and_invalid_offerings(self):
        self.offering["min_annual_revenue"] = 0
        self.profile["annual_revenue"] = None
        self.assertEqual(self.score()["rubric_breakdown"]["revenue_fit"], 100)
        for field, value in [("min_annual_revenue", None), ("required_tech_stack", [])]:
            with self.subTest(field=field):
                with patch.dict(self.offering, {field: value}):
                    self.assertEqual(self.score(), {
                        "score": None, "error": "Cannot score without a valid offering.",
                    })
        del self.offering["min_annual_revenue"]
        self.assertIsNone(self.score()["score"])

    def test_effective_stack_and_justification(self):
        self.profile["prospect_id"] = "LEAD-test"
        self.profile["tech_stack"] = list(self.offering["required_tech_stack"])

        def explain(messages):
            calculated = json.loads(messages[-1]["content"].split("Calculated scoring:\n")[1])
            matched = calculated["matched_technologies"]
            missing = calculated["missing_technologies"]
            return scoring.ScoreJustification(
                justification=f"Matched: {', '.join(matched)}. Missing: {', '.join(missing)}.",
            )

        self.llm.invoke.side_effect = explain
        with patch.object(scoring.data_service, "fetch_tech_stack", return_value=["Snowflake", "Tableau", "dbt"]):
            for _ in range(2):
                result = self.score()
                self.assertEqual(result["score"], 90)
                self.assertEqual(result["rubric_breakdown"]["tech_stack_match"], 75)
                self.assertEqual(result["justification"], "Matched: Snowflake, Tableau, dbt. Missing: Kafka.")
        self.assertEqual(self.profile["tech_stack"], self.offering["required_tech_stack"])

    def test_no_matching_technologies(self):
        self.profile["tech_stack"] = None
        self.assertEqual(self.score()["rubric_breakdown"]["tech_stack_match"], 0)
