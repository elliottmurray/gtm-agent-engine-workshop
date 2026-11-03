import json
import os
import re
import unittest
from types import SimpleNamespace
from unittest.mock import patch

with patch.dict(os.environ, {"OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "test-key")}):
    from .gtm_agent import run_agent, send_prospect_email


class PlainTextEmailTests(unittest.TestCase):
    def test_supplied_document_link_is_sent(self):
        body = "Instead of an attachment, view the pricing deck: https://example.com/pricing.pdf"
        result = send_prospect_email.func(
            prospect={"name": "Prospect", "email": "prospect@example.com"},
            subject="Pricing deck",
            body=body,
            runtime=SimpleNamespace(config={"metadata": {"user_id": "rep_sbrown"}}),
        )
        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["body"], body)
        self.assertEqual(result["to"], "prospect@example.com")


@unittest.skipUnless(os.environ.get("OPENAI_API_KEY"), "Requires model credentials")
class DocumentEmailFlowTests(unittest.TestCase):
    def run_request(self, request):
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}):
            return run_agent(request, user_id="rep_sbrown")

    def sent_emails(self, result):
        messages = []
        for message in result["messages"]:
            if message.type == "tool" and message.name == "send_prospect_email":
                payload = json.loads(message.content)
                if payload.get("status") == "sent":
                    messages.append(payload)
        return messages

    def test_document_without_link_waits_for_rep(self):
        result = self.run_request("Email LEAD-50003 to send over our pricing deck.")
        self.assertEqual(self.sent_emails(result), [])
        reply = result["reply"].lower()
        self.assertIn("link", reply)
        self.assertRegex(reply, r"(?:cannot|can't|unable|not able|don't have|does not support|doesn't support)")
        self.assertRegex(reply, r"attach|plain.text")
        self.assertRegex(reply, r"(?:please|provide|share|could you|can you|send me)")

    def test_document_with_link_sends_plain_text(self):
        link = "https://example.com/pricing.pdf"
        result = self.run_request(
            f"Email LEAD-50003 our pricing deck using this link: {link}"
        )
        emails = self.sent_emails(result)
        self.assertTrue(emails)
        for email in emails:
            self.assertIn(link, email["body"])
        for text in [email["body"] for email in emails] + [result["reply"]]:
            self.assertIsNone(re.search(
                r"attached (?:is|are)|(?:I(?:'ve| have)?|we(?:'ve| have)?) attached|"
                r"(?:deck|file|document) (?:is |was |has been )?attached|"
                r"with (?:the |an? )?attachment",
                text,
                re.IGNORECASE,
            ))


if __name__ == "__main__":
    unittest.main()
