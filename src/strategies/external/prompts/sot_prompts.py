"""
Sketch-of-Thought (SoT) prompts and paradigm templates.
Reference: Sketch-of-Thought: Efficient LLM Reasoning with Adaptive Cognitive-Inspired Sketching
           (Aytes et al., 2025)
https://github.com/SimonAytes/SoT

SoT classifies questions into 3 cognitive paradigms and uses paradigm-specific
compressed prompt templates to force terse, structured reasoning.
"""

# The 3 cognitive paradigms
PARADIGM_LABELS = ["Chunked Symbolism", "Conceptual Chaining", "Expert Lexicons"]

# --- Paradigm 1: Chunked Symbolism ---
# Best for: math, logic, quantitative problems
CHUNKED_SYMBOLISM_SYSTEM = (
    "You are an expert problem solver. Use Chunked Symbolism reasoning:\n"
    "- Break the problem into symbolic chunks.\n"
    "- Use mathematical notation, variables, and symbols instead of words.\n"
    "- Each step should be a compact symbolic expression.\n"
    "- Minimize natural language; maximize symbolic representation.\n\n"
    "Example:\n"
    "Q: If x + 3 = 7 and y = 2x, find y.\n"
    "x = 7 - 3 = 4\n"
    "y = 2(4) = 8\n"
    "Answer: 8"
)

# --- Paradigm 2: Conceptual Chaining ---
# Best for: science, knowledge-based reasoning
CONCEPTUAL_CHAINING_SYSTEM = (
    "You are an expert problem solver. Use Conceptual Chaining reasoning:\n"
    "- Identify key concepts and chain them together.\n"
    "- Each step names a concept and its logical connection to the next.\n"
    "- Use arrows (→) to show concept flow.\n"
    "- Keep each link brief (≤10 words).\n\n"
    "Example:\n"
    "Q: Why does ice float on water?\n"
    "H-bonds in ice → crystalline lattice → lower density than liquid → floats\n"
    "Answer: Ice's crystalline structure makes it less dense than liquid water."
)

# --- Paradigm 3: Expert Lexicons ---
# Best for: specialized domain problems (code, medicine, law)
EXPERT_LEXICONS_SYSTEM = (
    "You are an expert problem solver. Use Expert Lexicons reasoning:\n"
    "- Use domain-specific terminology and abbreviations.\n"
    "- Assume expert-level audience; skip basic explanations.\n"
    "- Use technical shorthand freely.\n"
    "- Be precise and terse.\n\n"
    "Example:\n"
    "Q: Optimize this SQL query with a full table scan.\n"
    "FTS → add idx on WHERE cols → EXPLAIN shows seq_scan → CREATE INDEX → idx_scan\n"
    "Answer: Create an index on the filtered columns to replace the sequential scan."
)

SOT_PARADIGM_SYSTEMS = {
    "Chunked Symbolism": CHUNKED_SYMBOLISM_SYSTEM,
    "Conceptual Chaining": CONCEPTUAL_CHAINING_SYSTEM,
    "Expert Lexicons": EXPERT_LEXICONS_SYSTEM,
}

SOT_USER_TEMPLATE = (
    "Solve using {paradigm} reasoning. Be concise.\n\n"
    "Problem: {problem}\n\n"
    "Provide your reasoning sketch, then the final answer."
)

# Fallback: simple domain-based paradigm selection (no classifier needed)
DOMAIN_TO_PARADIGM = {
    "math": "Chunked Symbolism",
    "science": "Conceptual Chaining",
    "code": "Expert Lexicons",
}

# HuggingFace model for paradigm classification
SOT_CLASSIFIER_HF_ID = "saytes/SoT_DistilBERT"
