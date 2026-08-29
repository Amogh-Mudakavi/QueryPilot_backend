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
    "grounding": "openai/gpt-oss-20b",
}


# Different starting positions so different tasks
# do not all begin with the same API key.
TASK_KEY_OFFSET = {
    "scope_guard": 0,
    "chitchat": 0,
    "planner": 0,
    "sql_generation": 1,
    "answer_generation": 1,
    "grounding": 2,
}


DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1024

MAX_RETRIES = 2
REQUEST_TIMEOUT = 30

DEFAULT_PROVIDER = "groq"