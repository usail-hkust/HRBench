"""
English prompt templates for all strategies in Hybrid Reasoning Benchmark.
All prompts are centralized here for easy management and reproducibility.
"""

# ============================================================
# Baseline Prompts
# ============================================================

# No special prompt needed for full_think / no_think — controlled via API params
# (enable_thinking=True/False or thinking_budget)

BUDGET_AWARE_SYSTEM_PROMPT = (
    "You are a helpful assistant. Solve the given problem. "
    "Adjust the depth of your reasoning according to the difficulty."
)

# ============================================================
# Training-Free: Prompt Tuning (model-specific)
# ============================================================

# Old generic PT prompt (kept for backward compat)
PROMPT_TUNING_SYSTEM = (
    "You are an expert problem solver with adaptive reasoning capabilities.\n\n"
    "Before solving each problem, assess its difficulty:\n"
    "- If the problem is **simple** (straightforward computation, basic concepts), "
    "solve it directly and concisely without extended reasoning.\n"
    "- If the problem is **complex** (multi-step reasoning, advanced concepts, tricky edge cases), "
    "think step by step in detail before giving your answer.\n\n"
    "You must decide the appropriate reasoning depth yourself based on the problem."
)

# --- Qwen3.5: guide <think> block depth ---
PROMPT_TUNING_SYSTEM_QWEN = (
    "You are an expert problem solver with adaptive reasoning.\n\n"
    "Before solving, assess the problem's difficulty and choose your reasoning depth:\n"
    "- For simple problems: Keep your <think> block empty or very brief. Answer directly.\n"
    "- For complex problems: Use your <think> block for thorough step-by-step reasoning.\n"
    "- For medium problems: Use your <think> block briefly for key observations only.\n\n"
    "You decide the appropriate depth based on the problem."
)

# --- gpt-oss: guide reasoning effort level ---
PROMPT_TUNING_SYSTEM_GPT_OSS = (
    "You are an expert problem solver. "
    "You may adjust your reasoning level between high, medium, and low "
    "based on problem complexity.\n\n"
    "- For simple problems: Use low reasoning effort. Minimal analysis, direct answer.\n"
    "- For complex problems: Use high reasoning effort. Thorough step-by-step analysis.\n"
    "- For medium problems: Use medium reasoning effort. Brief analysis on key steps.\n\n"
    "Assess each problem and choose the appropriate reasoning level yourself."
)

# --- Seed-OSS: guide thinking budget + reflection ---
PROMPT_TUNING_SYSTEM_SEED_OSS = (
    "You are an intelligent assistant with reflective reasoning ability. "
    "You may adjust your thinking budget based on problem complexity.\n\n"
    "- For simple problems: Set your thinking budget to 0 or minimal. "
    "Skip the thinking process and answer directly.\n"
    "- For complex problems: Allow a generous thinking budget (4096+ tokens). "
    "Think thoroughly and use <seed:cot_budget_reflect> to track your token usage.\n"
    "- For medium problems: Set a moderate thinking budget (512-1024 tokens). "
    "Think on key steps, reflect on progress, then answer.\n\n"
    "Example reflection during thinking:\n"
    "<seed:cot_budget_reflect>I have used 200 tokens, and there are 800 tokens "
    "remaining for use.</seed:cot_budget_reflect>\n\n"
    "Assess each problem and manage your thinking budget accordingly."
)

PROMPT_TUNING_USER_TEMPLATE = (
    "Solve the following problem. Decide whether it requires deep thinking or a direct answer.\n\n"
    "Problem: {problem}\n\n"
    "Provide your final answer."
)

# ============================================================
# Training-Free: Routing (model-specific judge prompts)
# ============================================================

# Old generic routing prompts (kept for backward compat)
ROUTING_JUDGE_SYSTEM = (
    "You are a problem difficulty classifier. "
    "Your task is to assess the difficulty of a given problem and decide the reasoning mode."
)

ROUTING_JUDGE_USER_TEMPLATE = (
    "Assess the difficulty of the following problem and respond with a JSON object.\n\n"
    "Problem: {problem}\n\n"
    "Respond with ONLY a JSON object in this format:\n"
    '{{"difficulty": "easy|medium|hard", "reasoning_mode": "nothink|think"}}\n\n'
    "Rules:\n"
    '- "easy": straightforward, single-step → "nothink"\n'
    '- "medium": multi-step but standard → "nothink"\n'
    '- "hard": complex, multi-step, tricky → "think"\n'
)

# --- Qwen3.5 Routing Judge: think / nothink / budget ---
ROUTING_JUDGE_SYSTEM_QWEN = "You are a problem difficulty classifier."

ROUTING_JUDGE_USER_QWEN = (
    "Assess the difficulty of the following problem and choose a reasoning mode.\n\n"
    "Problem: {problem}\n\n"
    "Available modes:\n"
    "1 - Think: Complex problem, enable full thinking with <think> block\n"
    "2 - NoThink: Simple problem, skip thinking, answer directly\n"
    "3 - Budget Think: Medium problem, think within a limited token budget\n\n"
    "Respond with ONLY a JSON object:\n"
    '{{"mode": "1" or "2" or "3", "budget": null or 1024 or 2048 or 4096}}\n\n'
    "Rules:\n"
    "- Mode 1: budget = null (unlimited thinking)\n"
    "- Mode 2: budget = null (no thinking)\n"
    "- Mode 3: budget = 1024 / 2048 / 4096\n"
)

# --- gpt-oss Routing Judge: reasoning effort high / medium / low ---
ROUTING_JUDGE_SYSTEM_GPT_OSS = "You are a problem difficulty classifier."

ROUTING_JUDGE_USER_GPT_OSS = (
    "Assess the difficulty of the following problem and choose a reasoning effort level.\n\n"
    "Problem: {problem}\n\n"
    "Available reasoning levels:\n"
    "1 - High: Complex problem, needs thorough step-by-step analysis\n"
    "2 - Low: Simple problem, minimal analysis, direct answer\n"
    "3 - Medium: Moderate problem, brief analysis on key steps\n\n"
    'Respond with ONLY a JSON object: {{"level": "high" or "medium" or "low"}}\n'
)

# --- Seed-OSS Routing Judge: full think / no think / budget ---
ROUTING_JUDGE_SYSTEM_SEED_OSS = "You are a problem difficulty classifier."

ROUTING_JUDGE_USER_SEED_OSS = (
    "Assess the difficulty of the following problem and choose a thinking strategy.\n\n"
    "Problem: {problem}\n\n"
    "Available thinking modes:\n"
    "1 - Full Think: Complex problem, unlimited thinking budget\n"
    "2 - No Think: Simple problem, skip thinking (budget = 0)\n"
    "3 - Budget Think: Medium problem, think within a fixed token budget\n\n"
    "Respond with ONLY a JSON object:\n"
    '{{"mode": "1" or "2" or "3", "budget": null or 512 or 1024 or 2048 or 4096}}\n\n'
    "Rules:\n"
    "- Mode 1: budget = null (unlimited)\n"
    "- Mode 2: budget = null (no thinking)\n"
    "- Mode 3: budget = 512 / 1024 / 2048 / 4096\n"
)

ROUTING_SOLVE_SYSTEM = "You are a helpful assistant. Solve the given problem carefully."

ROUTING_SOLVE_USER_TEMPLATE = "Problem: {problem}\n\nProvide your final answer."

# ============================================================
# Training-Free: Speculative Thinking
# ============================================================

# No special prompts — speculative strategies are controlled at token level
# The model starts with nothink mode and switches based on entropy or trigger words

SPECULATIVE_TRIGGER_WORDS = [
    # --- Hesitation / Uncertainty ---
    "wait",
    "hmm",
    "hm,",
    "hold on",
    "i'm not sure",
    "i am not sure",
    "not certain",
    "unclear",
    "confusing",

    # --- Self-correction / Backtracking ---
    "actually",
    "on second thought",
    "let me reconsider",
    "i made a mistake",
    "i made an error",
    "that's wrong",
    "that's incorrect",
    "that doesn't seem right",
    "this is wrong",
    "correction:",
    "i need to correct",
    "scratch that",
    "let me redo",
    "start over",
    "going back",

    # --- Re-examination / Verification ---
    "let me verify",
    "let me check",
    "let me re-examine",
    "let me recalculate",
    "double-check",
    "double check",
    "verify this",
    "verify that",
    "reconsider",
    "re-examine",
    "revisit",

    # --- Alternative approach ---
    "alternatively",
    "another approach",
    "another way",
    "different approach",
    "different method",
    "try a different",
    "let me try",
    "instead,",
    "perhaps",
    "maybe i should",

    # --- Deeper reasoning signals ---
    "think again",
    "think more carefully",
    "think step by step",
    "let me think",
    "need to think",
    "this requires",
    "this is tricky",
    "this is complex",
    "this is harder",
    "more carefully",
    "closer look",

    # --- Contradiction / Confusion ---
    "but that contradicts",
    "that contradicts",
    "this contradicts",
    "doesn't make sense",
    "does not make sense",
    "something is off",
    "something is wrong",
    "paradox",
    "inconsistent",

    # --- Explicit re-reasoning ---
    "recap",
    "summarize what we know",
    "let me summarize",
    "to be more precise",
    "more precisely",
    "to clarify",
    "in other words",
]

# ============================================================
# Math Answer Format Instructions
# ============================================================

MATH_ANSWER_INSTRUCTION = (
    "Put your final answer within \\boxed{{}}."
)

MATH_PROBLEM_TEMPLATE = (
    "{problem}\n\n" + MATH_ANSWER_INSTRUCTION
)

# ============================================================
# Code Answer Format Instructions
# ============================================================

CODE_ANSWER_INSTRUCTION = (
    "Write a Python solution. Read input from stdin and print output to stdout. "
    "Do not include any test code or examples. Only provide the solution code."
)

CODE_PROBLEM_TEMPLATE = (
    "{problem}\n\n" + CODE_ANSWER_INSTRUCTION
)

# ============================================================
# LLM-as-Judge Evaluation Prompt
# ============================================================

LLM_JUDGE_SYSTEM = (
    "You are an expert evaluator. Given a question, a reference answer, and a student's answer, "
    "determine if the student's answer is correct."
)

LLM_JUDGE_USER_TEMPLATE = (
    "Question:\n{problem}\n\n"
    "Reference Answer:\n{reference}\n\n"
    "Student's Answer:\n{response}\n\n"
    "Is the student's answer correct? Consider mathematical equivalence "
    "(e.g., 1/2 and 0.5 are equivalent, different forms of the same expression are equivalent).\n\n"
    'Respond with ONLY a JSON object: {{"correct": true}} or {{"correct": false}}'
)

# ============================================================
# Science (GPQA) Specific
# ============================================================

SCIENCE_PROBLEM_TEMPLATE = (
    "{problem}\n\n"
    "Provide your answer. If the problem is multiple-choice, state the correct option letter. "
    "Otherwise, provide a clear, concise answer."
)

# ============================================================
# SFT Data Construction Prompts
# ============================================================

SFT_MODE_SELECTION_SYSTEM = (
    "You are an adaptive reasoning assistant. For each problem, you must first decide "
    "your reasoning strategy, then solve the problem accordingly.\n\n"
    "Output format:\n"
    "[MODE: think] or [MODE: nothink]\n"
    "Then solve the problem."
)

# ============================================================
# RL Reward Description (for documentation)
# ============================================================

# ============================================================
# Strategy-specific prompt lookup (for training data + inference)
# ============================================================

BASELINE_SYSTEM = (
    "You are a helpful math assistant. "
    "Solve the problem step by step and provide your final answer."
)


def get_strategy_system_prompt(strategy: str, model_family: str) -> str:
    """
    Get the system prompt for a strategy + model family combination.

    This is the single source of truth used by BOTH:
      - Training data construction (sample_multimode, build_sft_data, etc.)
      - Inference strategies (training_based/strategies.py)

    Args:
        strategy: "pt" (prompt_tuning), "rt" (routing), or "baseline"
        model_family: "qwen", "gpt_oss", "seed_oss"

    Returns:
        The system prompt string.
    """
    # Baseline uses a generic prompt for all model families
    if strategy == "baseline":
        return BASELINE_SYSTEM

    _STRATEGY_PROMPT_MAP = {
        "pt": {
            "qwen": PROMPT_TUNING_SYSTEM_QWEN,
            "gpt_oss": PROMPT_TUNING_SYSTEM_GPT_OSS,
            "seed_oss": PROMPT_TUNING_SYSTEM_SEED_OSS,
        },
        "rt": {
            "qwen": ROUTING_SOLVE_SYSTEM,
            "gpt_oss": ROUTING_SOLVE_SYSTEM,
            "seed_oss": ROUTING_SOLVE_SYSTEM,
        },
    }
    if strategy not in _STRATEGY_PROMPT_MAP:
        raise ValueError(f"Unknown strategy: {strategy}. Must be 'pt', 'rt', or 'baseline'.")
    prompts = _STRATEGY_PROMPT_MAP[strategy]
    if model_family not in prompts:
        raise ValueError(
            f"Unknown model_family: {model_family}. "
            f"Must be one of: {list(prompts.keys())}"
        )
    return prompts[model_family]


RL_REWARD_DESCRIPTION = """
Reward function for RL (GRPO) training:
  score = alpha * accuracy + beta * efficiency * accuracy

Where:
  - accuracy: 1.0 if correct, 0.0 otherwise
  - efficiency: max(0, 1 - token_count / max_tokens)
  - Only reward efficiency when answer is correct
  - alpha (default 1.0): accuracy weight
  - beta (default 0.5): efficiency weight

Goal: Model learns to use nothink for easy problems (short + correct)
      and think for hard problems (longer but correct).
"""
