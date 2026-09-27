import json
import logging
from typing import Callable, Dict, List, Optional, Union
from pydantic import BaseModel, Field

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] ReActAgent - %(message)s"
)
logger = logging.getLogger("ReActAgent")


# =====================================================================
# 1. Action & Decision Schemas (Pydantic v2)
# =====================================================================

class ToolCall(BaseModel):
    tool_name: str = Field(description="Name of the tool to execute")
    tool_input: Dict[str, Union[str, int, float]] = Field(description="Dictionary of parameters for the tool")


class AgentStepDecision(BaseModel):
    thought: str = Field(description="The internal reasoning step of the agent")
    action: Optional[ToolCall] = Field(default=None, description="Tool to invoke if further action is needed")
    final_answer: Optional[str] = Field(default=None, description="Final answer to return to the user if task is complete")


# =====================================================================
# 2. Tool Registry
# =====================================================================

class ToolRegistry:
    """Manages available tools and their metadata for the ReAct Agent."""
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._schemas: Dict[str, str] = {}

    def register(self, name: str, description: str):
        """Decorator to register functions as tools."""
        def decorator(func: Callable):
            self._tools[name] = func
            self._schemas[name] = description
            return func
        return decorator

    def execute(self, tool_name: str, **kwargs) -> str:
        if tool_name not in self._tools:
            raise ValueError(f"Tool '{tool_name}' is not registered in the tool registry.")
        try:
            result = self._tools[tool_name](**kwargs)
            return str(result)
        except Exception as e:
            return f"Error executing tool '{tool_name}': {str(e)}"

    def get_tool_descriptions(self) -> str:
        return "\n".join([f"- {name}: {desc}" for name, desc in self._schemas.items()])


# =====================================================================
# 3. Define Concrete Tools
# =====================================================================

registry = ToolRegistry()

@registry.register(
    name="get_account_balance",
    description="Retrieves the current account balance for a given user_id. Input: {'user_id': str}"
)
def get_account_balance(user_id: str) -> str:
    # Mock database lookup
    balances = {"user_101": "$12,450.00", "user_102": "$320.50"}
    return balances.get(user_id, "$0.00 (User not found)")


@registry.register(
    name="calculate_tax",
    description="Calculates estimated tax on an amount. Input: {'amount': float, 'rate': float}"
)
def calculate_tax(amount: float, rate: float) -> str:
    tax = amount * rate
    return f"${tax:.2f}"


# =====================================================================
# 4. ReAct Core Engine
# =====================================================================

class ReActAgent:
    def __init__(self, tools: ToolRegistry, max_iterations: int = 5):
        self.tools = tools
        self.max_iterations = max_iterations

    def run(self, user_prompt: str) -> str:
        logger.info(f"Starting ReAct Planning for Task: '{user_prompt}'")
        
        # State tracking memory
        execution_history: List[str] = []
        
        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"--- Iteration {iteration}/{self.max_iterations} ---")
            
            # Step A: Generate Decision (Simulating LLM step-by-step reasoning based on state)
            decision = self._mock_llm_reasoning(user_prompt, execution_history, iteration)
            logger.info(f"THOUGHT: {decision.thought}")

            # Step B: Check if Agent has reached Final Answer
            if decision.final_answer:
                logger.info("Task completed successfully!")
                return decision.final_answer

            # Step C: Execute Action (Tool Call)
            if decision.action:
                tool_call = decision.action
                logger.info(f"ACTION: Executing '{tool_call.tool_name}' with args {tool_call.tool_input}")
                
                # Execute tool against registry
                observation = self.tools.execute(tool_call.tool_name, **tool_call.tool_input)
                logger.info(f"OBSERVATION: {observation}")
                
                # Record to history state for subsequent thoughts
                execution_history.append(
                    f"Iteration {iteration}: Thought: {decision.thought} | Action: {tool_call.tool_name}({tool_call.tool_input}) | Observation: {observation}"
                )
            else:
                logger.warning("No action or final answer provided by decision module. Halting.")
                break

        # Infinite loop / max iteration guardrail triggered
        logger.error("Max iterations reached without achieving a final answer.")
        return "Task failed: Exceeded maximum allowed planning iterations."

    def _mock_llm_reasoning(self, query: str, history: List[str], step: int) -> AgentStepDecision:
        """Simulates LLM reasoning based on current execution history state."""
        query_lower = query.lower()

        if "balance" in query_lower and "tax" in query_lower:
            if step == 1:
                return AgentStepDecision(
                    thought="The user wants to know their balance and calculate estimated tax at 15%. I must check the account balance first.",
                    action=ToolCall(tool_name="get_account_balance", tool_input={"user_id": "user_101"})
                )
            elif step == 2:
                return AgentStepDecision(
                    thought="I retrieved the balance ($12,450.00). Now I need to calculate 15% tax on 12450.0.",
                    action=ToolCall(tool_name="calculate_tax", tool_input={"amount": 12450.0, "rate": 0.15})
                )
            elif step == 3:
                return AgentStepDecision(
                    thought="I have both the balance ($12,450.00) and calculated tax ($1867.50). I can now synthesize the final response.",
                    final_answer="Your current account balance is $12,450.00. Based on a 15% tax rate, your estimated tax is $1,867.50."
                )

        elif "balance" in query_lower:
            if step == 1:
                return AgentStepDecision(
                    thought="The user is asking for their account balance. I will query the database tool.",
                    action=ToolCall(tool_name="get_account_balance", tool_input={"user_id": "user_101"})
                )
            elif step == 2:
                return AgentStepDecision(
                    thought="The balance has been retrieved. I will format the response for the user.",
                    final_answer="Your current account balance is $12,450.00."
                )

        # Fallback decision
        return AgentStepDecision(
            thought="I am unable to process this request with the available tools.",
            final_answer="I am sorry, but I do not have the required tools to fulfill this specific request."
        )


# =====================================================================
# 5. Pipeline Execution & Test Suite
# =====================================================================

if __name__ == "__main__":
    print("--- Starting Project 3: ReAct Planning Agent ---\n")

    agent = ReActAgent(tools=registry, max_iterations=5)

    test_prompts = [
        "What is the balance for user_101 and what would be the 15% tax on it?",
        "Check my account balance.",
        "What is the weather in Tokyo?"  # Tool unavailable test case
    ]

    for prompt in test_prompts:
        print(f"\nUser Request: '{prompt}'")
        final_result = agent.run(prompt)
        print(f"Result: {final_result}")
        print("=" * 70)
      
