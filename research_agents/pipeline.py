"""Shared multi-agent research pipeline, used by both the CLI and the web app.

Coordinates four subagents (planner, researcher, writer, critic) via the
Claude Agent SDK's Task-tool dispatch to turn a research question into a
sourced, reviewed Markdown report.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

from claude_agent_sdk import ClaudeAgentOptions, query
from claude_agent_sdk.types import AssistantMessage, TextBlock

from agents import AGENTS

OUTPUT_DIR = Path(__file__).parent / "output"

ORCHESTRATOR_PROMPT = """\
You are the lead coordinator of a multi-agent academic research team. Your job is to answer \
the user's research question thoroughly by orchestrating your teammates via the Task tool:

1. Call the "planner" subagent once with the research question to get 3-5 sub-questions.
2. Call the "researcher" subagent once PER sub-question to gather findings and sources for each.
3. Call the "writer" subagent once, giving it the original question and all researcher \
findings, to produce a first draft report.
4. Call the "critic" subagent once with the draft to get revision feedback.
5. If the critic did not respond "APPROVED", call the "writer" subagent one more time with \
the draft and the critique to produce a revised final report. Otherwise the first draft is final.
6. Save the final report to the exact file path you are given, using the Write tool.
7. Reply with a short (3-5 sentence) summary of the findings and the path to the saved report.

Do not skip steps and do not write report prose yourself - only the "writer" subagent should.
"""


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60] or "research-report"


async def stream_research(research_question: str) -> AsyncIterator[dict]:
    """Run the multi-agent pipeline, yielding progress events as they occur."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{slugify(research_question)}.md"

    options = ClaudeAgentOptions(
        system_prompt=ORCHESTRATOR_PROMPT,
        allowed_tools=["Task", "Write"],
        agents=AGENTS,
        permission_mode="acceptEdits",
        cwd=str(OUTPUT_DIR),
    )

    prompt = f"Research question: {research_question}\n\nSave the final report to: {report_path}"

    yield {"type": "info", "question": research_question, "report_path": str(report_path)}

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    if block.text.strip():
                        yield {"type": "text", "text": block.text}
                elif hasattr(block, "name"):
                    tool_name = block.name
                    if tool_name == "Task":
                        subagent = getattr(block, "input", {}).get("subagent_type", "?")
                        yield {"type": "dispatch", "subagent": subagent}
                    else:
                        yield {"type": "tool", "name": tool_name}

    yield {
        "type": "done",
        "report_path": str(report_path),
        "success": report_path.exists(),
    }
