import logging
from datetime import date
from typing import Any

from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

# ── Config ────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL = "qwen2.5:7b" # mistral was poor for tool use, so using qwen2.5:7b instead
TODAY = "2026-05-05"

# ── Logger ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ToolLogger(BaseCallbackHandler):
    def on_tool_start(self, serialized: dict, input_str: str, **kwargs: Any) -> None:
        name = serialized.get("name", "unknown")
        logger.info("[TOOL START] %s | input: %s", name, input_str)

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        logger.info("[TOOL END]   output: %s", output)

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        logger.error("[TOOL ERROR] %s", error)

# ── Fake Data ─────────────────────────────────────────────────────
FAKE_CALENDAR = {
    TODAY: [
        {"time": "09:00", "title": "Team Standup Meeting"},
        {"time": "14:00", "title": "Project Review"},
        {"time": "18:00", "title": "Dinner Appointment"},
    ],
    "2026-05-06": [
        {"time": "10:00", "title": "Code Review"},
    ],
}

FAKE_INBOX = [
    {"id": 1, "from": "010-1234-5678", "message": "Can we reschedule tomorrow's meeting?", "read": False},
    {"id": 2, "from": "010-9876-5432", "message": "Want to grab lunch together?", "read": True},
    {"id": 3, "from": "010-5555-0000", "message": "When can you send the report?", "read": False},
]

SENT_EMAILS = []
SAVED_EVENTS = []

# ── Tools ─────────────────────────────────────────────────────────
@tool
def get_today_schedule() -> dict:
    """Retrieve today's calendar schedule."""
    today = TODAY # str(date.today())
    events = FAKE_CALENDAR.get(today, [])
    return {
        "date": today,
        "events": events if events else [],
        "message": f"You have {len(events)} event(s) today." if events else "No events scheduled for today.",
    }


@tool
def save_schedule(date: str, time: str, title: str) -> dict:
    """Save a new event to the calendar. date: YYYY-MM-DD, time: HH:MM, title: event title."""
    SAVED_EVENTS.append({"date": date, "time": time, "title": title})
    if date not in FAKE_CALENDAR:
        FAKE_CALENDAR[date] = []
    FAKE_CALENDAR[date].append({"time": time, "title": title})
    return {"success": True, "message": f"Event saved: [{date} {time}] {title}"}


@tool
def send_email(to: str, subject: str, body: str) -> dict:
    """Send an email to the specified recipient."""
    SENT_EMAILS.append({"to": to, "subject": subject, "body": body})
    return {"success": True, "message": f"Email successfully sent to '{to}'. Subject: {subject}"}


@tool
def read_sms(unread_only: bool = False) -> dict:
    """Read the list of received SMS messages. Set unread_only=True to return only unread messages."""
    messages = [m for m in FAKE_INBOX if not m["read"]] if unread_only else FAKE_INBOX
    return {"total": len(messages), "messages": messages}


# ── LLM ──────────────────────────────────────────────────────────
llm = ChatOllama(model=MODEL, base_url=OLLAMA_BASE_URL)

# ── Agent ─────────────────────────────────────────────────────────
tools = [get_today_schedule, save_schedule, send_email, read_sms]

agent = create_agent(llm, tools, system_prompt="You are a helpful personal assistant.")


def ask_assistant(user_message: str) -> str:
    result = agent.invoke(
        {"messages": [("human", user_message)]},
        config={"callbacks": [ToolLogger()]},
    )
    return result["messages"][-1].content


# ── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    questions = [
        "What's on my schedule today?",
        "Add a 'Weekly Report' event tomorrow at 10 AM.",
        "Send an email to Kim Chulsoo (kim@example.com) with subject 'Meeting Schedule' and body 'There is a meeting tomorrow at 2 PM.'",
        "Check my unread text messages.",
    ]

    for q in questions:
        print(f"\n>>>>>>> User: {q}")
        answer = ask_assistant(q)
        print(f"\n<<<<<<< Assistant: {answer}")
        print("─" * 50)