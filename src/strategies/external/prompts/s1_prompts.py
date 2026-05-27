"""
S1 Budget Forcing prompts and constants.
Reference: s1: Simple test-time scaling (Muennighoff et al., ICLR 2025)
https://github.com/simplescaling/s1

Budget forcing works by manipulating stop tokens during vLLM generation:
1. Ignore the model's end-of-thinking stop token
2. Append "Wait" to force the model to continue reasoning
3. Control total thinking length via max_tokens
"""

# Token to inject when forcing the model to continue thinking
WAIT_TOKEN = "\nWait"

# Budget levels mapped to max thinking tokens
# These control how many tokens the model spends in its <think> block
S1_BUDGET_MAP = {
    "low": 1024,     # minimal thinking
    "medium": 4096,  # moderate thinking
    "high": 16384,   # extensive thinking
}

# Number of times to inject "Wait" before giving up
# (safety limit to prevent infinite loops)
MAX_WAIT_INJECTIONS = 10
