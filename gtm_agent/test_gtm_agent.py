"""Offline scoring and request-context tests."""

import json
import os
import unittest
from unittest.mock import Mock, patch

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from pydantic import Field

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import gtm_agent as agent


class ScriptedModel(GenericFakeChatModel):
    requests: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, **kwargs):
        self.requests.append(messages)
        return super()._generate(messages, **kwargs)


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"LANGSMITH_TRACING": "false"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.cache = patch.object(agent.data_service, "_PROFILES", {})
        self.cache.start()
        self.addCleanup(self.cache.stop)
        self.score = agent.ProspectScore(
            score=65, justification="Model assessment",
            rubric_breakdown=agent.RubricBreakdown(
                revenue_fit=70, tech_stack_match=50, segment_fit=75,
            ),
        )
        self.scorer = Mock()
        self.scorer.invoke.return_value = self.score
        scoring_patch = patch.object(agent, "_scoring_llm", self.scorer)
        scoring_patch.start()
        self.addCleanup(scoring_patch.stop)

    def run_script(self, responses, request, user_id="rep_amills"):
        model = ScriptedModel(messages=iter(responses))
        graph = agent.create_deep_agent(
            model=model,
            tools=[agent.lookup_offering, agent.build_prospect_profile,
                   agent.get_prospect, agent.send_prospect_email,
                   agent.score_prospect, agent.update_prospect_info,
                   agent.get_current_rep],
            system_prompt=agent.SYSTEM_PROMPT,
        )
        with patch.object(agent, "gtm_agent", graph):
            result = agent.run_agent(request, user_id=user_id)
        return result, model

    def tool_call(self, name, args, call_id="call-1"):
        return AIMessage(content="", tool_calls=[
            {"name": name, "args": args, "id": call_id},
        ])

    def test_score_loads_records_from_ids(self):
        result = agent.score_prospect.invoke({
            "prospect_id": "LEAD-71001", "offering_id": "OFFER-10005",
        })
        self.assertEqual(result, self.score.model_dump())
        user = self.scorer.invoke.call_args.args[0][1]["content"]
        offering, profile = user.removeprefix("Offering:\n").split("\n\nProspect profile:\n")
        self.assertEqual(json.loads(offering), agent.data_service.get_offering("OFFER-10005"))
        self.assertEqual(json.loads(profile)["prospect_id"], "LEAD-71001")
        self.assertEqual(json.loads(profile)["tech_stack"],
                         agent.data_service.fetch_tech_stack("LEAD-71001"))
        self.assertIsNotNone(agent.data_service.get_profile_from_db("LEAD-71001")["prospect_profile"])

    def test_score_refreshes_cached_tech_stack(self):
        agent.build_prospect_profile.invoke({"prospect_id": "LEAD-71001"})
        with patch.object(agent.data_service, "fetch_tech_stack", return_value=["Fresh technology"]):
            agent.score_prospect.invoke({"prospect_id": "LEAD-71001", "offering_id": "OFFER-10005"})
        user = self.scorer.invoke.call_args.args[0][1]["content"]
        profile = json.loads(user.split("\n\nProspect profile:\n")[1])
        self.assertEqual(profile["tech_stack"], ["Fresh technology"])

    def test_missing_prospect_is_recoverable(self):
        result = agent.score_prospect.invoke({
            "prospect_id": "unknown", "offering_id": "OFFER-10005",
        })
        self.assertIsNone(result["score"])
        self.assertFalse(result["found"])
        self.assertIsNone(result["prospect_profile"])
        self.assertTrue(result["error"])
        self.scorer.invoke.assert_not_called()

    def test_missing_or_incomplete_offering_is_recoverable(self):
        valid = agent.data_service.get_offering("OFFER-10005")
        offerings = [None, {}] + [
            {key: value for key, value in valid.items() if key != missing}
            for missing in ("description", "required_tech_stack", "min_annual_revenue")
        ]
        for offering in offerings:
            with self.subTest(offering=offering), patch.object(
                agent.data_service, "get_offering", return_value=offering,
            ):
                result = agent.score_prospect.invoke({
                    "prospect_id": "LEAD-71001", "offering_id": "unknown",
                })
                self.assertIsNone(result["score"])
                self.assertEqual(result["found"], offering is not None)
                self.assertTrue(result["error"])
        self.scorer.invoke.assert_not_called()

    def test_score_workflow_uses_only_ids_without_rep_lookup(self):
        args = {"prospect_id": "LEAD-71001", "offering_id": "OFFER-10005"}
        result, model = self.run_script([
            self.tool_call("score_prospect", args), AIMessage(content="Scored"),
        ], "Score LEAD-71001 against OFFER-10005.")
        self.assertEqual(result["reply"], "Scored")
        calls = [call for message in result["messages"] if isinstance(message, AIMessage)
                 for call in message.tool_calls]
        self.assertEqual([(call["name"], call["args"]) for call in calls], [("score_prospect", args)])
        outputs = [json.loads(message.content) for message in result["messages"]
                   if isinstance(message, ToolMessage)]
        self.assertEqual(outputs, [self.score.model_dump()])
        self.assertEqual(len(model.requests), 2)

    def test_model_can_recover_from_missing_prospect(self):
        result, model = self.run_script([
            self.tool_call("score_prospect", {
                "prospect_id": "unknown", "offering_id": "OFFER-10005",
            }),
            self.tool_call("score_prospect", {
                "prospect_id": "LEAD-71001", "offering_id": "OFFER-10005",
            }, "call-2"),
            AIMessage(content="Recovered"),
        ], "Score the prospect.")
        outputs = [json.loads(message.content) for message in result["messages"]
                   if isinstance(message, ToolMessage)]
        self.assertFalse(outputs[0]["found"])
        self.assertIsNone(outputs[0]["score"])
        self.assertEqual(outputs[1], self.score.model_dump())
        self.assertEqual(result["reply"], "Recovered")
        self.assertEqual(len(model.requests), 3)
        self.scorer.invoke.assert_called_once()

    def test_rep_context_is_request_specific_without_tool_turn(self):
        for user_id in ("rep_amills", "rep_jchen", "unknown"):
            with self.subTest(user_id=user_id):
                result, model = self.run_script([AIMessage(content="Done")], "Hello", user_id)
                contexts = [message.text for message in model.requests[0]
                            if isinstance(message, SystemMessage) and message.text.startswith("Signed-in rep: ")]
                self.assertEqual(len(contexts), 1)
                context = json.loads(contexts[0].removeprefix("Signed-in rep: "))
                rep = agent.data_service.get_rep(user_id)
                self.assertEqual(context, {"rep": rep, "found": rep is not None})
                self.assertEqual(len(model.requests), 1)
                self.assertEqual(result["reply"], "Done")
        self.scorer.invoke.assert_not_called()

    def test_email_workflow_keeps_model_draft_and_default_sender(self):
        prospect = {"name": "Prospect", "email": "prospect@example.com"}
        result, model = self.run_script([
            self.tool_call("send_prospect_email", {
                "prospect": prospect, "subject": "Model subject", "body": "Model draft",
            }), AIMessage(content="Sent"),
        ], "Email the prospect.", "rep_jchen")
        outputs = [json.loads(message.content) for message in result["messages"]
                   if isinstance(message, ToolMessage)]
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0]["status"], "sent")
        self.assertEqual(outputs[0]["body"], "Model draft")
        self.assertEqual(outputs[0]["subject"], "Model subject")
        self.assertEqual(outputs[0]["to"], prospect["email"])
        self.assertEqual(outputs[0]["from"], agent.data_service.get_rep("rep_jchen")["email"])
        self.assertEqual(len(model.requests), 2)
        self.scorer.invoke.assert_not_called()

    def test_optional_rep_lookup_preserves_found_flags(self):
        for user_id in ("rep_amills", "unknown"):
            with self.subTest(user_id=user_id):
                result, _ = self.run_script([
                    self.tool_call("get_current_rep", {}), AIMessage(content="Done"),
                ], "Look up the current rep.", user_id)
                output = next(json.loads(message.content) for message in result["messages"]
                              if isinstance(message, ToolMessage))
                rep = agent.data_service.get_rep(user_id)
                self.assertEqual(output, {"rep": rep, "found": rep is not None})


if __name__ == "__main__":
    unittest.main()
