import json
import logging
import re
from typing import Any, Dict, List, Type, TypeVar
from pydantic import BaseModel, Field, ValidationError, field_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("StructuredOutputAgent")

T = TypeVar("T", bound=BaseModel)

# =====================================================================
# 1. Target Schema Definitions (Pydantic v2)
# =====================================================================

class CustomerActionItem(BaseModel):
    action_type: str = Field(description="Action category: REFUND, SUPPORT, ESCALATE, or IGNORE")
    priority: int = Field(description="Priority ranking from 1 (Low) to 5 (Urgent)")
    summary: str = Field(description="Concise sentence detailing required action")

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v: int) -> int:
        if not (1 <= v <= 5):
            raise ValueError(f"Priority must be between 1 and 5. Got {v}.")
        return v

    @field_validator("action_type")
    @classmethod
    def validate_action_type(cls, v: str) -> str:
        valid_types = {"REFUND", "SUPPORT", "ESCALATE", "IGNORE"}
        if v.upper() not in valid_types:
            raise ValueError(f"action_type must be one of {valid_types}. Got '{v}'.")
        return v.upper()


class CustomerSupportTicket(BaseModel):
    ticket_id: str = Field(description="Format: TICKET-XXXX where X are numbers")
    customer_email: str = Field(description="Valid corporate email address")
    urgency_score: float = Field(description="Score between 0.0 and 1.0")
    action_items: List[CustomerActionItem] = Field(description="Required action items")

    @field_validator("ticket_id")
    @classmethod
    def validate_ticket_id(cls, v: str) -> str:
        if not re.match(r"^TICKET-\d{4}$", v):
            raise ValueError("ticket_id must follow pattern 'TICKET-XXXX' (e.g., TICKET-1024)")
        return v

    @field_validator("urgency_score")
    @classmethod
    def validate_urgency(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"urgency_score must be between 0.0 and 1.0. Got {v}.")
        return v

# =====================================================================
# 2. Mock Provider
# =====================================================================

class MockLLMProvider:
    """Simulates an LLM that makes structural errors on its first try."""
    def __init__(self):
        self.attempt_counter = 0

    def generate(self, prompt: str, schema_json: str) -> str:
        self.attempt_counter += 1
        logger.info(f"LLM Generation Attempt #{self.attempt_counter}")

        if self.attempt_counter == 1:
            return json.dumps({
                "ticket_id": "INVALID-ID",
                "customer_email": "john.doe@acme.com",
                "urgency_score": 0.85,
                "action_items": [
                    {
                        "action_type": "GIVE_DISCOUNT",
                        "priority": 6,
                        "summary": "Process $50 credit"
                    }
                ]
            })

        return json.dumps({
            "ticket_id": "TICKET-4821",
            "customer_email": "john.doe@acme.com",
            "urgency_score": 0.85,
            "action_items": [
                {
                    "action_type": "REFUND",
                    "priority": 4,
                    "summary": "Process $50 refund for double-billed transaction"
                }
            ]
        })

# =====================================================================
# 3. Agent Engine & Self-Correction Retry Loop
# =====================================================================

class StructuredOutputAgent:
    def __init__(self, llm_provider: MockLLMProvider, max_retries: int = 3):
        self.llm = llm_provider
        self.max_retries = max_retries

    def run(self, user_prompt: str, response_model: Type[T]) -> T:
        schema = response_model.model_json_schema()
        schema_str = json.dumps(schema, indent=2)

        messages = [
            {"role": "system", "content": f"You are a structured extraction engine. You MUST respond with a valid JSON object matching this schema:\n{schema_str}"},
            {"role": "user", "content": user_prompt}
        ]

        for attempt in range(1, self.max_retries + 1):
            logger.info(f"Executing step (Attempt {attempt}/{self.max_retries})")
            
            prompt_context = self._build_prompt_str(messages)
            raw_response = self.llm.generate(prompt_context, schema_str)

            try:
                parsed_json = self._clean_and_parse_json(raw_response)
            except ValueError as parse_err:
                logger.warning(f"JSON Parse Error on attempt {attempt}: {str(parse_err)}")
                messages.append({"role": "assistant", "content": raw_response})
                messages.append({
                    "role": "user",
                    "content": f"Your response was not valid JSON. Parse Error: {str(parse_err)}. Please fix and return ONLY valid JSON."
                })
                continue

            try:
                validated_obj = response_model.model_validate(parsed_json)
                logger.info("Schema validation successful!")
                return validated_obj

            except ValidationError as val_err:
                error_feedback = self._format_validation_error(val_err)
                logger.warning(f"Validation Failure on attempt {attempt}:\n{error_feedback}")

                messages.append({"role": "assistant", "content": raw_response})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your output failed schema validation with errors:\n{error_feedback}\n"
                        "Please correct your output and strictly conform to the expected format."
                    )
                })

        raise RuntimeError(f"Agent failed to produce valid output after {self.max_retries} attempts.")

    @staticmethod
    def _clean_and_parse_json(raw_text: str) -> Dict[str, Any]:
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)
        return json.loads(cleaned)

    @staticmethod
    def _format_validation_error(err: ValidationError) -> str:
        formatted_errors = []
        for e in err.errors():
            loc = " -> ".join([str(p) for p in e["loc"]])
            msg = e["msg"]
            formatted_errors.append(f"- Location [{loc}]: {msg}")
        return "\n".join(formatted_errors)

    @staticmethod
    def _build_prompt_str(messages: List[Dict[str, str]]) -> str:
        return "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])

# =====================================================================
# 4. Entrypoint
# =====================================================================

if __name__ == "__main__":
    print("--- Running Project 1: Structured Output Agent ---\n")

    agent = StructuredOutputAgent(llm_provider=MockLLMProvider(), max_retries=3)

    user_input = (
        "Process customer email from John Doe (john.doe@acme.com). "
        "He complains about double billing on invoice 1024 and requests a $50 refund immediately."
    )

    result: CustomerSupportTicket = agent.run(
        user_prompt=user_input,
        response_model=CustomerSupportTicket
    )

    print("\n--- Final Validated Output ---")
    print(f"Ticket ID:      {result.ticket_id}")
    print(f"Customer Email: {result.customer_email}")
    print(f"Urgency Score:  {result.urgency_score}")
    print(f"Actions ({len(result.action_items)}):")
    for item in result.action_items:
        print(f"  - [{item.action_type}] Priority {item.priority}: {item.summary}")
