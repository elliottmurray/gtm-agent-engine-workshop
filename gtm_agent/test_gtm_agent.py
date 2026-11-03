import importlib
import json
import os
import unittest
import uuid
from unittest.mock import patch

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI


class RunRecorder(BaseCallbackHandler):
    def __init__(self):
        self.roots = []
        self.tools = []

    def on_chain_start(
        self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs
    ):
        if parent_run_id is None:
            self.roots.append((run_id, kwargs))

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.tools.append((serialized["name"], kwargs["metadata"]))


class GTMAgentTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "offline-test-key",
                "LANGSMITH_TRACING": "false",
            },
        )
        env.start()
        self.addCleanup(env.stop)
        with patch("dotenv.load_dotenv"):
            self.module = importlib.import_module("gtm_agent.gtm_agent")
        self.recorder = RunRecorder()
        graph = self.module.gtm_agent.with_config(callbacks=[self.recorder])
        graph_patch = patch.object(self.module, "gtm_agent", graph)
        graph_patch.start()
        self.addCleanup(graph_patch.stop)
        self.calls = []

    def run_request(self, request, steps):
        responses = iter(steps)

        def generate(messages, **kwargs):
            self.calls.append((messages, kwargs["tools"]))
            step = next(responses)
            message = (
                AIMessage(content="Done.")
                if step is None
                else AIMessage(
                    content="",
                    tool_calls=[
                        {"name": name, "args": args, "id": str(i)}
                        for i, (name, args) in enumerate(step)
                    ],
                )
            )
            return ChatResult(generations=[ChatGeneration(message=message)])

        with patch.object(ChatOpenAI, "_generate", side_effect=generate):
            result = self.module.run_agent(
                request,
                user_id="rep_jchen",
                environment="test",
                thread_id="test-thread",
            )
        self.assertEqual(result["reply"], "Done.")
        self.assertEqual(result["messages"][-1].content, result["reply"])
        self.assertEqual(self.recorder.roots[0][0], uuid.UUID(result["run_id"]))
        self.assertEqual(self.recorder.roots[0][1]["name"], "GTM Assistant")
        expected_metadata = {
            "user_id": "rep_jchen",
            "environment": "test",
            "thread_id": "test-thread",
            "request_intent": self.module.classify_intent(request),
        }
        for metadata in [self.recorder.roots[0][1]["metadata"]] + [
            metadata for _, metadata in self.recorder.tools
        ]:
            for key, value in expected_metadata.items():
                self.assertEqual(metadata[key], value)
        for messages, tools in self.calls:
            self.assertEqual(
                [m.content for m in messages if m.type == "system"],
                [self.module.SYSTEM_PROMPT],
            )
            self.assertEqual(
                {t["function"]["name"] for t in tools},
                {
                    "lookup_offering",
                    "build_prospect_profile",
                    "get_prospect",
                    "send_prospect_email",
                    "score_prospect",
                    "update_prospect_info",
                    "get_current_rep",
                },
            )
            for tool in tools:
                self.assertNotIn(
                    "runtime", tool["function"]["parameters"]["properties"]
                )
        return {
            m.name: json.loads(m.content)
            for m in result["messages"]
            if isinstance(m, ToolMessage)
        }

    def test_email_uses_signed_in_rep_and_only_gtm_tools(self):
        prospect = {"name": "Test Prospect", "email": "prospect@example.com"}
        results = self.run_request(
            "Email LEAD-71001 to arrange a demo.",
            [
                [("get_current_rep", {})],
                [("get_prospect", {"prospect_id": "LEAD-71001"})],
                [
                    (
                        "send_prospect_email",
                        {"prospect": prospect, "subject": "Demo", "body": "Hello"},
                    )
                ],
                None,
            ],
        )
        rep = self.module.data_service.get_rep("rep_jchen")
        self.assertEqual(results["get_current_rep"], {"rep": rep, "found": True})
        self.assertTrue(results["get_prospect"]["found"])
        self.assertEqual(results["send_prospect_email"]["status"], "sent")
        self.assertEqual(results["send_prospect_email"]["from"], rep["email"])
        self.assertEqual(results["send_prospect_email"]["to"], prospect["email"])

    def test_profile_update_and_scoring_tools_complete(self):
        score = self.module.ProspectScore(
            score=80,
            justification="Good fit",
            rubric_breakdown={
                "revenue_fit": 80,
                "tech_stack_match": 80,
                "segment_fit": 80,
            },
        )
        offering = self.module.data_service.get_offering("OFFER-10005")
        with (
            patch.dict(self.module.data_service._PROFILES, {}, clear=True),
            patch.object(
                self.module,
                "_scoring_llm",
                **{"invoke.return_value": score},
            ),
        ):
            results = self.run_request(
                "Update and score LEAD-71001 against OFFER-10005.",
                [
                    [("get_current_rep", {})],
                    [
                        ("lookup_offering", {"offering_id": "OFFER-10005"}),
                        ("build_prospect_profile", {"prospect_id": "LEAD-71001"}),
                        (
                            "update_prospect_info",
                            {"prospect_id": "LEAD-71001", "technology": "Kafka"},
                        ),
                    ],
                    [
                        (
                            "score_prospect",
                            {
                                "prospect_profile": {"prospect_id": "LEAD-71001"},
                                "offering": offering,
                            },
                        )
                    ],
                    None,
                ],
            )
        self.assertEqual(results["lookup_offering"]["offering"], offering)
        self.assertTrue(results["build_prospect_profile"]["found"])
        self.assertTrue(results["update_prospect_info"]["updated"])
        self.assertEqual(results["score_prospect"], score.model_dump())


if __name__ == "__main__":
    unittest.main()
