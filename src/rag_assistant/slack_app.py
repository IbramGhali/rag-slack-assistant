"""Slack integration (Bolt for Python, Socket Mode).

Message handling and Block Kit rendering are plain functions so they can be tested without
Slack; `create_app` only wires them to Bolt events. Needs the `slack` extra plus
SLACK_BOT_TOKEN and SLACK_APP_TOKEN.
"""

from __future__ import annotations

import re
from typing import Any

from rag_assistant.answer import Answer, Assistant
from rag_assistant.feedback import FeedbackStore

MENTION = re.compile(r"<@[A-Z0-9]+>")
HELPFUL, NOT_HELPFUL = "feedback_helpful", "feedback_not_helpful"


def clean_question(text: str) -> str:
    return MENTION.sub("", text).strip()


def answer_blocks(answer: Answer, answer_id: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = [
        {"type": "section", "text": {"type": "mrkdwn", "text": answer.text}}
    ]
    if answer.cited:
        lines = []
        for n in answer.cited:
            chunk = answer.sources[n - 1].chunk
            note = " (superseded version)" if chunk.superseded else ""
            lines.append(f"[{n}] *{chunk.title}*, {chunk.section}{note}")
        blocks.append(
            {"type": "context", "elements": [{"type": "mrkdwn", "text": "\n".join(lines)}]}
        )
    elif answer.refused:
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "No policy covers this clearly enough to answer. "
                        "Try rephrasing, or ask the policy owner.",
                    }
                ],
            }
        )
    blocks.append(
        {
            "type": "actions",
            "block_id": f"feedback:{answer_id}",
            "elements": [
                {
                    "type": "button",
                    "action_id": HELPFUL,
                    "value": answer_id,
                    "text": {"type": "plain_text", "text": "Helpful"},
                },
                {
                    "type": "button",
                    "action_id": NOT_HELPFUL,
                    "value": answer_id,
                    "text": {"type": "plain_text", "text": "Not helpful"},
                },
            ],
        }
    )
    return blocks


def handle_question(
    assistant: Assistant, store: FeedbackStore, text: str, user_id: str, channel: str
) -> dict[str, Any]:
    """Returns the chat.postMessage payload for a question."""
    answer = assistant.ask(clean_question(text))
    answer_id = store.log_answer(answer, channel, user_id)
    return {"channel": channel, "text": answer.text, "blocks": answer_blocks(answer, answer_id)}


def handle_feedback(store: FeedbackStore, body: dict[str, Any]) -> str:
    action = body["actions"][0]
    helpful = action["action_id"] == HELPFUL
    store.record(action["value"], body["user"]["id"], helpful)
    return "Thanks, noted." if helpful else "Thanks. This answer has been flagged for review."


def create_app(
    assistant: Assistant, store: FeedbackStore, bot_token: str, **bolt_options: Any
) -> Any:
    from slack_bolt import App

    app = App(token=bot_token, **bolt_options)

    def respond(event: dict[str, Any], say: Any) -> None:
        payload = handle_question(
            assistant, store, event.get("text", ""), event.get("user", ""), event["channel"]
        )
        say(text=payload["text"], blocks=payload["blocks"], thread_ts=event.get("ts"))

    @app.event("app_mention")
    def on_mention(event: dict[str, Any], say: Any) -> None:
        respond(event, say)

    @app.event("message")
    def on_dm(event: dict[str, Any], say: Any) -> None:
        if event.get("channel_type") == "im" and not event.get("bot_id"):
            respond(event, say)

    @app.action(HELPFUL)
    @app.action(NOT_HELPFUL)
    def on_feedback(ack: Any, body: dict[str, Any], client: Any) -> None:
        ack()
        message = handle_feedback(store, body)
        client.chat_postEphemeral(
            channel=body["channel"]["id"], user=body["user"]["id"], text=message
        )

    return app
