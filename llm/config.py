"""QueryPilot LLM configuration.

Three things live here:

* the Groq keys and per-model rate limits (the real provider constraints),
* one ``TaskProfile`` per logical task (model, output cap, timeout, reasoning),
* retry / cooldown policy.

There is deliberately NO request-wide token budget. Every task owns its
generation settings, and nothing that happened earlier in the same user
message can lower them. ``input_hint`` is an observability threshold only: the
provider logs a warning when a prompt exceeds it, it never truncates or refuses.
"""

from dataclasses import dataclass

from django.conf import settings

GROQ_API_KEYS = [
    settings.GROQ_API_KEY_1,
    settings.GROQ_API_KEY_2,
    settings.GROQ_API_KEY_3,
]

if not all(GROQ_API_KEYS):
    raise ValueError("All Groq API keys must be configured.")

MODEL_TPM_LIMITS = {
    "openai/gpt-oss-120b": 8000,
    "openai/gpt-oss-20b": 8000,
    "qwen/qwen3.8-27b": 8000,
}
DEFAULT_TPM_LIMIT = 8000

TPM_ADMISSION_RATIO = 0.90
TPM_WINDOW_SECONDS = 60.0

MAX_ADMISSION_WAIT_SECONDS = 20.0
ADMISSION_MAX_BYPASS = 3
CHARS_PER_TOKEN_ESTIMATE = 3.5
MESSAGE_OVERHEAD_TOKENS = 4

MODEL_20B = "openai/gpt-oss-20b"
MODEL_120B = "openai/gpt-oss-120b"
MODEL_QWEN_27B = "qwen/qwen3.8-27b"

# One source of truth for task completion ceilings. Qwen reasoning tokens are
# included in max_tokens, so these caps leave room for both reasoning and output.
PLANNER_MAX_COMPLETION_TOKENS = 1800
TABLE_SELECTION_MAX_COMPLETION_TOKENS = 1500
SQL_GENERATION_MAX_COMPLETION_TOKENS = 1200
ANSWER_OUTPUT_TOKEN_CAPS = {
    "normal": 2000,
    "arithmetic": 2600,
    "multi_step": 3200,
}
# Maximum task ceiling also bounds the single retry after a length finish.
ANSWER_MAX_OUTPUT_TOKENS = 4000
ANSWER_RETRY_MULTIPLIER = 2

REQUEST_TIMEOUT = 20.0
DEFAULT_TEMPERATURE = 0.0


@dataclass(frozen=True)
class TaskProfile:
    """Generation settings owned by one logical task.

    model            which tier runs it (the tier decides the quota pool)
    max_tokens       maximum completion ceiling for this task, including any
                     reasoning tokens counted by the provider.
    timeout          per-attempt HTTP timeout in seconds
    reasoning_effort "low" / "medium" / "high", or None for the model default
    input_hint       prompt size above which an observability warning is logged.
    """

    model: str
    max_tokens: int
    timeout: float = REQUEST_TIMEOUT
    reasoning_effort: str | None = None
    input_hint: int | None = None


TASK_PROFILES: dict[str, TaskProfile] = {
    "scope_guard": TaskProfile(MODEL_20B, max_tokens=500, input_hint=1200),
    "chitchat": TaskProfile(
        MODEL_20B, max_tokens=500, reasoning_effort="low", input_hint=600
    ),
    "title": TaskProfile(
        MODEL_20B, max_tokens=200, reasoning_effort="low", input_hint=200
    ),
    "sql_generation": TaskProfile(
        MODEL_QWEN_27B,
        max_tokens=SQL_GENERATION_MAX_COMPLETION_TOKENS,
        input_hint=3000,
    ),
    "planner": TaskProfile(
        MODEL_QWEN_27B,
        max_tokens=PLANNER_MAX_COMPLETION_TOKENS,
        input_hint=3000,
    ),
    "table_selection": TaskProfile(
        MODEL_QWEN_27B,
        max_tokens=TABLE_SELECTION_MAX_COMPLETION_TOKENS,
        reasoning_effort="low",
        input_hint=2000,
    ),
    "answer_generation": TaskProfile(
        MODEL_120B,
        max_tokens=ANSWER_MAX_OUTPUT_TOKENS,
        timeout=40.0,
        reasoning_effort="low",
        input_hint=3000,
    ),
}

TASK_MODELS = {name: profile.model for name, profile in TASK_PROFILES.items()}

MAX_ATTEMPTS = 3
RETRY_LIMITS = {
    "rate_limit": 2,  # hop to another key; the 429'd bucket is cooled
    "authentication": 2,  # disable that key, hop to another
    "timeout": 1,  # a slow model rarely recovers; one more try on another key
    "connection": 2,
    "provider_5xx": 2,
}

RETRY_BACKOFF_SECONDS = 1.0

# Used when a 429 carries no retry hint.
DEFAULT_COOLDOWN_SECONDS = 20.0
MAX_COOLDOWN_SECONDS = 3600.0
# A key that returned 401/403 is taken out of rotation for this long.
AUTH_COOLDOWN_SECONDS = 300.0