#!/usr/bin/env python3
"""Entry point for the multi-agent academic research system.

Coordinates four subagents (planner, researcher, writer, critic) via the
Claude Agent SDK's Task-tool dispatch to turn a research question into a
sourced, reviewed Markdown report.
"""

import asyncio
import re
import sys
from datetime import datetime
from pathlib import Path

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


async def run(research_question: str) -> None:
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

    print(f"Research question: {research_question}")
    print(f"Report will be saved to: {report_path}\n")
    print("--- Running multi-agent research pipeline ---\n")

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    if block.text.strip():
                        print(block.text)
                elif hasattr(block, "name"):
                    tool_name = block.name
                    if tool_name == "Task":
                        subagent = getattr(block, "input", {}).get("subagent_type", "?")
                        print(f"[dispatch -> {subagent} subagent]")
                    else:
                        print(f"[tool: {tool_name}]")

    print("\n--- Done ---")
    if report_path.exists():
        print(f"Final report saved at: {report_path}")
    else:
        print("Warning: expected report file was not found - check the log above.")


def main() -> None:
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Enter your research question: ").strip()

    if not question:
        print("A research question is required.", file=sys.stderr)
        sys.exit(1)

    asyncio.run(run(question))


if __name__ == "__main__":
    main()
