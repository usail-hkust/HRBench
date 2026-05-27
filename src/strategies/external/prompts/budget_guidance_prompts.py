"""
Budget Guidance prompts (prompt-only simplified version).
Reference: Steering LLM Thinking with Budget Guidance (UMass, 2025)
https://github.com/UMass-Embodied-AGI/BudgetGuidance

The full method uses a forked transformers library with token_budget parameter.
This is a simplified prompt-only version that instructs the model to respect
an explicit token budget in the prompt.
"""

BUDGET_GUIDANCE_SYSTEM = (
    "You are a precise problem solver. You MUST complete your reasoning "
    "within the specified token budget. Plan your approach based on the "
    "budget constraint:\n"
    "- Small budget: Skip intermediate steps, answer directly.\n"
    "- Medium budget: Cover key reasoning steps only.\n"
    "- Large budget: Full step-by-step reasoning allowed.\n\n"
    "Stay within the budget. Quality over quantity."
)

BUDGET_GUIDANCE_USER_TEMPLATE = (
    "**Token Budget: {budget} tokens**\n\n"
    "Complete your reasoning and answer within approximately {budget} tokens.\n\n"
    "Problem: {problem}\n\n"
    "Solve within the budget."
)

# Budget levels (aligned with the original paper's budget settings)
BUDGET_GUIDANCE_MAP = {
    "low": 128,
    "medium": 512,
    "high": 2048,
}
