"""
TALE-EP (Token-Budget-Aware LLM Reasoning) prompts.
Reference: Token-Budget-Aware LLM Reasoning (Han et al., ACL 2025 Findings)
https://github.com/GeniusHTX/TALE

TALE-EP is the training-free variant (Explicit Planning):
The prompt asks the model to first estimate the token budget needed,
then reason within that budget.
"""

TALE_EP_SYSTEM = (
    "You are a precise problem solver who manages reasoning length efficiently.\n\n"
    "Before solving any problem, you must:\n"
    "1. **Estimate** the token budget needed for this problem "
    "(e.g., simple=100, medium=300, hard=800+ tokens).\n"
    "2. **Solve** the problem within your estimated budget.\n"
    "3. **Monitor** your token usage and wrap up if approaching the budget.\n\n"
    "Format your response as:\n"
    "[Budget: <N> tokens]\n"
    "<your reasoning and solution within ~N tokens>\n\n"
    "Be concise when the problem is simple. Only use extensive reasoning for truly complex problems."
)

TALE_EP_USER_TEMPLATE = (
    "Estimate your token budget, then solve this problem within that budget.\n\n"
    "Problem: {problem}\n\n"
    "Start with [Budget: <N> tokens], then solve."
)

# Budget levels for TALE with explicit budget override
# When using TALE with a fixed budget, we prepend budget instruction
TALE_FIXED_BUDGET_TEMPLATE = (
    "You MUST solve this problem using approximately {budget} tokens of reasoning.\n"
    "Be concise and focused. Do not exceed {budget} tokens.\n\n"
    "Problem: {problem}\n\n"
    "Solve within ~{budget} tokens."
)

TALE_BUDGET_MAP = {
    "low": 100,
    "medium": 500,
    "high": 1500,
}
