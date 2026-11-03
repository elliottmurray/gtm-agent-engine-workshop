import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

with patch("dotenv.load_dotenv"), patch.dict(
    os.environ,
    {"OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY") or "unit-test", "LANGSMITH_TRACING": "false"},
):
    from gtm_agent import gtm_agent as module


class SendProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.prospect = {"prospect_id": "LEAD-TEST", "name": "Prospect", "email": "prospect@example.com"}
        self.rep = {"name": "Rep", "email": "rep@example.com"}
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": "rep-test"}})

    def send(self, prospect):
        return module.send_prospect_email.func(
            prospect, "Demo invite", "Please join our demo.", self.runtime, self.rep
        )

    def test_disqualified_source_blocks_even_if_caller_claims_eligible(self):
        with patch.object(module.data_service, "get_prospect_record", return_value={"disqualified": True}), patch.object(
            module.uuid, "uuid4", side_effect=AssertionError("Blocked email reached message construction")
        ):
            result = self.send({**self.prospect, "disqualified": False})
        self.assertEqual(result["status"], "blocked")
        self.assertIn("disqualified", result["error"])
        self.assertNotIn("message_id", result)

    def test_missing_id_or_source_blocks_before_message_construction(self):
        for prospect in ({"email": "prospect@example.com"}, {**self.prospect, "prospect_id": ""}, self.prospect):
            with self.subTest(prospect=prospect), patch.object(
                module.data_service, "get_prospect_record", return_value=None
            ), patch.object(module.uuid, "uuid4", side_effect=AssertionError("Blocked email reached message construction")):
                result = self.send(prospect)
            self.assertEqual(result["status"], "blocked")
            self.assertIn("eligibility could not be verified", result["error"])
            self.assertIn("source prospect record" if prospect.get("prospect_id") else "prospect ID", result["error"])
            self.assertNotIn("message_id", result)

    def test_eligible_prospect_sends_with_existing_message_fields(self):
        with patch.object(module.data_service, "get_prospect_record", return_value={"disqualified": False}):
            result = self.send(self.prospect)
        self.assertEqual(result["status"], "sent")
        self.assertTrue(result["message_id"].startswith("msg-"))
        self.assertEqual(result["to"], self.prospect["email"])
        self.assertEqual(result["to_name"], self.prospect["name"])
        self.assertEqual(result["from"], self.rep["email"])
        self.assertEqual(result["from_name"], self.rep["name"])
        self.assertEqual(result["subject"], "Demo invite")
        self.assertEqual(result["body"], "Please join our demo.")

    def test_eligible_prospect_without_email_still_fails(self):
        with patch.object(module.data_service, "get_prospect_record", return_value={"disqualified": False}):
            result = self.send({"prospect_id": "LEAD-TEST"})
        self.assertEqual(result, {"status": "failed", "error": "Prospect record has no email address."})

    def test_eligible_prospect_defaults_to_signed_in_rep(self):
        with patch.object(module.data_service, "get_prospect_record", return_value={"disqualified": False}), patch.object(
            module.data_service, "get_rep", return_value=self.rep
        ):
            result = module.send_prospect_email.func(self.prospect, "Demo invite", "Join us.", self.runtime)
        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["from"], self.rep["email"])

    @unittest.skipUnless(os.environ.get("OPENAI_API_KEY"), "Requires OpenAI credentials for agent behavior")
    def test_agent_reports_blocked_reason_without_retry(self):
        for prospect, record, reason in (
            (self.prospect, {"disqualified": True}, "disqualified"),
            (self.prospect, None, "record"),
            ({"email": "prospect@example.com"}, None, "id"),
        ):
            with self.subTest(reason=reason), patch.object(
                module.data_service, "get_prospect_record", return_value=record
            ):
                blocked = self.send(prospect)
                messages = [
                    HumanMessage(content="Email this prospect a demo invite."),
                    AIMessage(content="", tool_calls=[{"name": "get_current_rep", "args": {}, "id": "rep"}]),
                    ToolMessage(content='{"rep": {"name": "Rep", "email": "rep@example.com"}, "found": true}', tool_call_id="rep"),
                    AIMessage(content="", tool_calls=[{
                        "name": "send_prospect_email", "id": "send",
                        "args": {"prospect": prospect, "subject": "Demo invite", "body": "Join us."},
                    }]),
                    ToolMessage(content=json.dumps(blocked), tool_call_id="send"),
                ]
                with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}):
                    result = module.gtm_agent.invoke({"messages": messages}, config={"recursion_limit": 10})
            subsequent = result["messages"][len(messages):]
            sends = [call for message in subsequent for call in getattr(message, "tool_calls", [])
                     if call["name"] == "send_prospect_email"]
            self.assertEqual(sends, [])
            reply = result["messages"][-1].content.lower()
            self.assertRegex(reply, r"not sent|wasn't sent|didn't send|could not send|couldn't send|unable to send")
            self.assertIn(reason, reply)


if __name__ == "__main__":
    unittest.main()
