# Project 1: Structured Output Agent

A production-grade implementation of a **Structured Output Agent** in Python using **Pydantic v2** and an **Instructor-style auto-retry self-correction loop**.

## Key Features
- **Schema Enforcement:** Dynamically generates standard JSON schemas from nested Pydantic models.
- **Runtime Validation:** Intercepts structure violations, bad enum values, and field bounds failures.
- **Self-Healing Loop:** Automatically captures validation tracebacks and feeds re-prompts back to the model for correction.
- **Zero Third-Party Agent Framework Dependencies:** Pure Python + Pydantic v2 implementation.

## How to Run

1. Install dependencies:
   ```pip install -r requirements.txt```
2. Execute the script
   ```python main.py```