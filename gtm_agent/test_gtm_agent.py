import copy
import importlib
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch


with (
    patch("dotenv.load_dotenv"),
    patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}),
    patch("langchain_openai.ChatOpenAI"),
    patch("deepagents.create_deep_agent"),
):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class ProspectPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.prospect_id = "LEAD-TEST"
        self.record = {
            "name": "Test Prospect",
            "email": "prospect@example.com",
            "annual_revenue": 70000000,
            "billing_qualification": {"tax_id": "private-tax-id"},
            "engagement_history": [{"channel": "Webinar"}],
            "account_details": [{"account": "Example", "industry": "Software"}],
            "tech_stack": ["AWS"],
        }
        self.offering = {
            "required_tech_stack": ["AWS"],
            "min_annual_revenue": 50000000,
            "description": "Cloud platform",
        }
        self.enterContext(patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}))
        self.enterContext(patch.dict(agent.data_service.PROSPECTS,
                                     {self.prospect_id: self.record}, clear=True))
        self.enterContext(patch.dict(agent.data_service._PROFILES, {}, clear=True))

    def test_contact_omits_billing_and_preserves_email_fields(self):
        result = agent.get_prospect.invoke({"prospect_id": self.prospect_id})
        self.assertTrue(result["found"])
        self.assertEqual(result["prospect"], {
            "prospect_id": self.prospect_id,
            "name": self.record["name"],
            "email": self.record["email"],
            "annual_revenue": self.record["annual_revenue"],
        })
        sent = agent.send_prospect_email.func(
            result["prospect"], "Hello", "Let's talk",
            SimpleNamespace(config={}),
            from_rep={"name": "Rep", "email": "rep@example.com"},
        )
        self.assertEqual(sent["status"], "sent")
        self.assertEqual(sent["to"], self.record["email"])
        self.assertEqual(sent["to_name"], self.record["name"])

    def test_new_profile_omits_billing_before_caching(self):
        result = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        expected = {k: v for k, v in self.record.items() if k != "billing_qualification"}
        expected["prospect_id"] = self.prospect_id
        self.assertEqual(result, {"prospect_profile": expected, "found": True})
        self.assertEqual(agent.data_service._PROFILES[self.prospect_id], expected)
        self.assertIn("billing_qualification", self.record)

    def test_cached_profile_omits_billing(self):
        cached = {"prospect_id": self.prospect_id, **copy.deepcopy(self.record)}
        agent.data_service._PROFILES[self.prospect_id] = cached
        result = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        expected = {k: v for k, v in cached.items() if k != "billing_qualification"}
        self.assertEqual(result, {"prospect_profile": expected, "found": True})
        self.assertIn("billing_qualification", cached)

    def test_scoring_strips_billing_from_incoming_profiles(self):
        for has_id in (True, False):
            with self.subTest(has_id=has_id):
                profile = copy.deepcopy(self.record)
                if has_id:
                    profile.update(prospect_id=self.prospect_id, tech_stack=["stale"])
                expected = {k: v for k, v in profile.items() if k != "billing_qualification"}
                expected["tech_stack"] = self.record["tech_stack"]
                score = {"score": 90, "justification": "Strong fit"}

                def score_messages(messages):
                    serialized = json.dumps(messages)
                    self.assertNotIn("billing_qualification", serialized)
                    self.assertNotIn("private-tax-id", serialized)
                    user = messages[1]["content"]
                    offering, prospect = user.removeprefix("Offering:\n").split(
                        "\n\nProspect profile:\n"
                    )
                    self.assertEqual(json.loads(offering), self.offering)
                    self.assertEqual(json.loads(prospect), expected)
                    return SimpleNamespace(model_dump=lambda: score)

                with patch.object(agent._scoring_llm, "invoke", side_effect=score_messages):
                    result = agent.score_prospect.invoke({
                        "prospect_profile": profile, "offering": self.offering,
                    })
                self.assertEqual(result, score)
                self.assertIn("billing_qualification", profile)

    def test_built_and_cached_profiles_can_be_scored(self):
        for _ in range(2):
            profile = agent.build_prospect_profile.invoke({
                "prospect_id": self.prospect_id,
            })["prospect_profile"]
            with patch.object(agent._scoring_llm, "invoke") as invoke:
                invoke.return_value.model_dump.return_value = {"score": 90}
                result = agent.score_prospect.invoke({
                    "prospect_profile": profile, "offering": self.offering,
                })
            self.assertNotIn("billing_qualification", json.dumps(profile))
            self.assertNotIn("billing_qualification", json.dumps(invoke.call_args.args))
            self.assertEqual(result, {"score": 90})


if __name__ == "__main__":
    unittest.main()
