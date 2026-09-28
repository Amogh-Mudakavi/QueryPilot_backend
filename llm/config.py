from django.conf import settings


GROQ_API_KEYS = [
    settings.GROQ_API_KEY_1,
    settings.GROQ_API_KEY_2,
    settings.GROQ_API_KEY_3,
]

if not all(GROQ_API_KEYS):
    raise ValueError(
        "All Groq API keys must be configured."
    )


TASK_MODELS = {
    "scope_guard": "openai/gpt-oss-20b",
    "chitchat": "openai/gpt-oss-20b",
    "planner": "openai/gpt-oss-120b",
    "sql_generation": "openai/gpt-oss-20b",
    "answer_generation": "openai/gpt-oss-120b",
}


DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1024

# Total number of attempts, including the first request.
MAX_ATTEMPTS = 3

# Maximum time allowed for ONE provider attempt. Worst case per LLM call is
# roughly MAX_ATTEMPTS * REQUEST_TIMEOUT plus backoff (~65s at 3 x 20s).
REQUEST_TIMEOUT = 20.0

# Base retry delay for non-429 failures (timeouts, network errors).
# Backoff sequence: 1s -> 2s
RETRY_BACKOFF_SECONDS = 1.0

DEFAULT_COOLDOWN_SECONDS = 60.0

DEFAULT_PROVIDER = "groq"