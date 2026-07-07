#!/usr/bin/env python3
"""CLI entry point for the multi-agent academic research system."""

import asyncio
import sys

from pipeline import stream_research


async def run(research_question: str) -> None:
    print(f"Research question: {research_question}\n")

    async for event in stream_research(research_question):
        etype = event["type"]
        if etype == "info":
            print(f"Report will be saved to: {event['report_path']}\n")
            print("--- Running multi-agent research pipeline ---\n")
        elif etype == "text":
            print(event["text"])
        elif etype == "dispatch":
            print(f"[dispatch -> {event['subagent']} subagent]")
        elif etype == "tool":
            print(f"[tool: {event['name']}]")
        elif etype == "done":
            print("\n--- Done ---")
            if event["success"]:
                print(f"Final report saved at: {event['report_path']}")
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
