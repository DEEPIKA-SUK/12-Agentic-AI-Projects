# Project 3: ReAct Planning Agent
An autonomous reasoning and tool-execution agent implementing the ReAct (Reasoning + Acting) framework with dynamic tool registration, state memory tracking, and infinite-loop execution guardrails.

---

## Features
- **ReAct Execution Loop:** Implements the core **Thought $\rightarrow$ Action $\rightarrow$ Observation** loop to solve multi-step tasks.
- **Dynamic Tool Registry:** Flexible function decorator pattern (`@registry.register`) that exposes custom Python functions as available tools to the agent.
- **Circuit Breaker Guardrail:** Enforces `max_iterations` limits to prevent infinite execution loops during tool or reasoning failures.
- **Structured Decision Contracts:** Type-safe Pydantic v2 models enforcing clean separation between intermediate tool calls (`ToolCall`) and terminal responses (`final_answer`).
---

## Architecture & Workflow
```
 ┌────────────────┐
 │ User Prompt    │
 └───────┬────────┘
         │
         ▼
 ┌─────────────────────────────────────────────────────────────┐
 │                      ReAct Agent Loop                       │
 │                                                             │
 │  1. THOUGHT ──────► 2. ACTION ──────► 3. OBSERVATION        │
 │  (Analyze State)    (Call Tool)       (Capture Tool Output) │
 │         ▲                                    │              │
 │         └────────────────────────────────────┘              │
 └───────────────────────┬─────────────────────────────────────┘
                         │
             Terminal / Final Answer
                         │
                         ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ Final User Response / Output                                │
 └─────────────────────────────────────────────────────────────┘
```
---

## Project Structure
```
project-3/
├── requirements.txt   # Core Python dependencies
├── main.py            # ReAct agent engine, tool registry, and test suite
└── README.md          # Project documentation
```

---
## Commands
```
pip install -r requirements.txt
python main.py
```
---
## Output
```
User Request: 'What is the balance for user_101 and what would be the 15% tax on it?'
2026-09-27 18:42:00 [INFO] ReActAgent - Starting ReAct Planning for Task: 'What is the balance for user_101 and what would be the 15% tax on it?'
2026-09-27 18:42:00 [INFO] ReActAgent - --- Iteration 1/5 ---
2026-09-27 18:42:00 [INFO] ReActAgent - THOUGHT: The user wants to know their balance and calculate estimated tax at 15%. I must check the account balance first.
2026-09-27 18:42:00 [INFO] ReActAgent - ACTION: Executing 'get_account_balance' with args {'user_id': 'user_101'}
2026-09-27 18:42:00 [INFO] ReActAgent - OBSERVATION: $12,450.00
2026-09-27 18:42:00 [INFO] ReActAgent - --- Iteration 2/5 ---
2026-09-27 18:42:00 [INFO] ReActAgent - THOUGHT: I retrieved the balance ($12,450.00). Now I need to calculate 15% tax on 12450.0.
2026-09-27 18:42:00 [INFO] ReActAgent - ACTION: Executing 'calculate_tax' with args {'amount': 12450.0, 'rate': 0.15}
2026-09-27 18:42:00 [INFO] ReActAgent - OBSERVATION: $1867.50
2026-09-27 18:42:00 [INFO] ReActAgent - --- Iteration 3/5 ---
2026-09-27 18:42:00 [INFO] ReActAgent - THOUGHT: I have both the balance ($12,450.00) and calculated tax ($1867.50). I can now synthesize the final response.
2026-09-27 18:42:00 [INFO] ReActAgent - Task completed successfully!
Result: Your current account balance is $12,450.00. Based on a 15% tax rate, your estimated tax is $1,867.50.
```

