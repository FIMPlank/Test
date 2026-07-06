# Multi-Agent Academic Research System

A small research-assistant pipeline built on the [Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/python).
A lead orchestrator agent coordinates four specialist subagents to turn a research
question into a sourced, reviewed Markdown report:

```
planner   -> splits the question into 3-5 focused sub-questions
researcher -> one call per sub-question; searches the web and summarizes findings + sources
writer    -> synthesizes all findings into a structured draft report
critic    -> reviews the draft; if not approved, writer revises once more
```

The orchestrator saves the final report to `research_agents/output/` and prints a
live log of which subagent is running as the pipeline executes.

## Prerequisites

- Python 3.10+
- Node.js (the Python SDK shells out to the Claude Code CLI)
- An `ANTHROPIC_API_KEY` environment variable

```bash
npm install -g @anthropic-ai/claude-code
pip install -r research_agents/requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...   # or copy .env.example to .env and fill it in
```

## Run from the command line

```bash
python research_agents/orchestrator.py "What are the current approaches to improving sample efficiency in reinforcement learning?"
```

Omit the argument and it will prompt you for a question interactively.

## Run from VS Code

1. Open this repo in VS Code with the Python extension installed.
2. Open the **Run and Debug** panel and select **"Run Multi-Agent Research System"**.
3. Press Run/F5 - you'll be prompted for a research question in the command palette input box.
4. Watch the pipeline log in the integrated terminal; the finished report path is printed at the end.

> If your Python extension predates the `debugpy` debug type, change `"type": "debugpy"` to
> `"type": "python"` in `.vscode/launch.json`.

## Output

Reports are written to `research_agents/output/<timestamp>-<slug>.md` (git-ignored).
