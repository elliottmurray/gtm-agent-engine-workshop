import importlib
import json
import unittest
from copy import deepcopy
from unittest.mock import patch

from gtm_agent import data_service

with patch("langchain_openai.ChatOpenAI"), patch("deepagents.create_deep_agent"):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class UpdateProspectInfoTests(unittest.TestCase):
    def setUp(self):
        self.records = patch.dict(data_service.PROSPECTS, deepcopy(data_service.PROSPECTS))
        self.profiles = patch.dict(data_service._PROFILES, {}, clear=True)
        self.records.start()
        self.profiles.start()
        self.addCleanup(self.records.stop)
        self.addCleanup(self.profiles.stop)
        self.prospect_id = "LEAD-39002"

    def test_update_persists_stack_and_preserves_response(self):
        original = list(data_service.fetch_tech_stack(self.prospect_id))

        result = data_service.update_prospect_info(self.prospect_id, "Terraform")

        self.assertEqual(result, {
            "updated": True,
            "found": True,
            "tech_stack": original + ["Terraform"],
        })
        self.assertEqual(data_service.fetch_tech_stack(self.prospect_id), result["tech_stack"])

    def test_update_rebuilds_cached_profile(self):
        original = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        self.assertNotIn("Terraform", original["prospect_profile"]["tech_stack"])

        data_service.update_prospect_info(self.prospect_id, "Terraform")
        rebuilt = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})

        self.assertTrue(rebuilt["found"])
        self.assertIn("Terraform", rebuilt["prospect_profile"]["tech_stack"])

    def test_unknown_prospect_does_not_report_update(self):
        self.assertEqual(
            data_service.update_prospect_info("UNKNOWN", "Terraform"),
            {"updated": False, "found": False},
        )
        self.assertIsNone(data_service.get_prospect_record("UNKNOWN"))

    def test_existing_technology_is_not_duplicated(self):
        original = list(data_service.fetch_tech_stack(self.prospect_id))

        result = data_service.update_prospect_info(self.prospect_id, original[0])

        self.assertTrue(result["updated"])
        self.assertEqual(data_service.fetch_tech_stack(self.prospect_id), original)

    def test_scoring_uses_persisted_stack_even_with_old_profile(self):
        profile = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        offering = data_service.get_offering("OFFER-10004")
        scoring_stacks = []

        def score(messages):
            user = messages[1]["content"]
            supplied_profile = json.loads(user.split("\n\nProspect profile:\n", 1)[1])
            stack = supplied_profile["tech_stack"]
            scoring_stacks.append(stack)
            missing = sorted(set(offering["required_tech_stack"]) - set(stack))
            return agent.ProspectScore(
                score=100 if not missing else 75,
                justification="Missing technologies: " + ", ".join(missing),
                rubric_breakdown=agent.RubricBreakdown(
                    revenue_fit=100, tech_stack_match=100 if not missing else 75,
                    segment_fit=100,
                ),
            )

        inputs = {"prospect_profile": profile["prospect_profile"], "offering": offering}
        with patch.object(agent, "_scoring_llm") as llm:
            llm.invoke.side_effect = score
            before = agent.score_prospect.invoke(inputs)
            agent.update_prospect_info.invoke({
                "prospect_id": self.prospect_id, "technology": "Terraform",
            })
            self.assertIn("Terraform", data_service.fetch_tech_stack(self.prospect_id))
            after = agent.score_prospect.invoke(inputs)

        self.assertNotIn("Terraform", scoring_stacks[0])
        self.assertIn("Terraform", scoring_stacks[1])
        self.assertIn("Terraform", before["justification"])
        self.assertNotIn("Terraform", after["justification"])
        self.assertGreater(after["score"], before["score"])


if __name__ == "__main__":
    unittest.main()
