"""Config module for QueryPilot."""

from django.conf import settings

GROQ_API_KEYS = [
    settings.GROQ_API_KEY_1,
    settings.GROQ_API_KEY_2,
    settings.GROQ_API_KEY_3,
]

if not all(GROQ_API_KEYS):
    raise ValueError("All Groq API keys must be configured.")


TASK_MODELS = {
    "scope_guard": "openai/gpt-oss-20b",
    "chitchat": "openai/gpt-oss-20b",
    "planner": "openai/gpt-oss-120b",
    "table_selection": "openai/gpt-oss-120b",
    "sql_generation": "openai/gpt-oss-20b",
    "answer_generation": "openai/gpt-oss-120b",
}


MODEL_TPM_LIMITS = {
    "openai/gpt-oss-120b": 8000,
    "openai/gpt-oss-20b": 8000,
}
DEFAULT_TPM_LIMIT = 8000


TPM_ADMISSION_RATIO = 0.90
TPM_WINDOW_SECONDS = 60.0


MAX_ADMISSION_WAIT_SECONDS = 20.0


TASK_TOKEN_BUDGETS = {
    "scope_guard": {"input": 1200, "completion": 400},
    "chitchat": {"input": 500, "completion": 300},
    "planner": {"input": 3000, "completion": 1000},
    "table_selection": {"input": 3500, "completion": 800},
    "sql_generation": {"input": 3000, "completion": 1200},
    "answer_generation": {"input": 3500, "completion": 2000},
}

DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1024


MAX_ATTEMPTS = 3


REQUEST_TIMEOUT = 20.0


RETRY_BACKOFF_SECONDS = 1.0

DEFAULT_COOLDOWN_SECONDS = 60.0

DEFAULT_PROVIDER = "groq"
