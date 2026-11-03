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

# One source of truth for task completion ceilings. Reasoning tokens are
# included in max_tokens, so these caps leave room for reasoning and output.
PLANNER_MAX_COMPLETION_TOKENS = 3400
TABLE_SELECTION_MAX_COMPLETION_TOKENS = 2200
SQL_GENERATION_MAX_COMPLETION_TOKENS = 2600
ANSWER_OUTPUT_TOKEN_CAPS = {
    "normal": 2000,
    "arithmetic": 2600,
    "multi_step": 3200,
}
# Maximum task ceiling also bounds the single retry after a length finish.
ANSWER_MAX_OUTPUT_TOKENS = 4000
# Deprecated: the application no longer estimates TPM headroom, so these two
# are no longer used by answer.py. Kept only so stray imports do not break.
ANSWER_MIN_OUTPUT_TOKENS = 256
ANSWER_TPM_SAFETY_MARGIN_TOKENS = 256
ANSWER_RETRY_MULTIPLIER = 2
# Provider-guided fitting: when Groq rejects a request because input +
# max_tokens exceeds the TPM limit, the provider lowers max_tokens by exactly
# the excess Groq reports (plus this margin), but only if what is left is still
# a usable completion for a reasoning model. Otherwise the error is raised.
MIN_USEFUL_COMPLETION_TOKENS = 1500
FIT_MARGIN_TOKENS = 50

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
    # The 120B tasks are deliberately isolated onto separate API-key lanes.
    # See TASK_API_KEY_INDEX below; an idle lane is not borrowed by another task.
    "scope_guard": TaskProfile(MODEL_20B, max_tokens=500, input_hint=1200),
    "chitchat": TaskProfile(
        MODEL_20B, max_tokens=500, reasoning_effort="low", input_hint=600
    ),
    "title": TaskProfile(
        MODEL_20B, max_tokens=200, reasoning_effort="low", input_hint=200
    ),
    "sql_generation": TaskProfile(
        MODEL_120B,
        max_tokens=SQL_GENERATION_MAX_COMPLETION_TOKENS,
        timeout=REQUEST_TIMEOUT,
        reasoning_effort="low",
        input_hint=3000,
    ),
    "planner": TaskProfile(
        MODEL_120B,
        max_tokens=PLANNER_MAX_COMPLETION_TOKENS,
        timeout=40.0,
        reasoning_effort="low",
        input_hint=3000,
    ),
    # Table selection shares the structured-data lane with SQL generation.
    # Its model fallback stays on this same key.
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

# Zero-based indices into GROQ_API_KEYS (GROQ_API_KEY_1 is index 0).
# Every configured task is pinned to exactly one independent API key.
# Model fallback retains the same task assignment and never borrows another key.
TASK_API_KEY_INDEX: dict[str, int] = {
    # Key 1: planning and scope/control tasks.
    "planner": 0,
    "scope_guard": 0,
    "chitchat": 0,
    "title": 0,
    # Key 2: structured-data discovery and SQL.
    "table_selection": 1,
    "sql_generation": 1,
    # Key 3: user-facing answer generation.
    "answer_generation": 2,
}

if len(GROQ_API_KEYS) < 3:
    raise ValueError(
        "The 120B task lanes require GROQ_API_KEY_1, GROQ_API_KEY_2, and GROQ_API_KEY_3."
    )
if any(index < 0 or index >= len(GROQ_API_KEYS) for index in TASK_API_KEY_INDEX.values()):
    raise ValueError("TASK_API_KEY_INDEX references a missing Groq API key.")

TASK_MODELS = {name: profile.model for name, profile in TASK_PROFILES.items()}

if set(TASK_API_KEY_INDEX) != set(TASK_PROFILES):
    missing = sorted(set(TASK_PROFILES) - set(TASK_API_KEY_INDEX))
    unknown = sorted(set(TASK_API_KEY_INDEX) - set(TASK_PROFILES))
    raise ValueError(
        f"Every LLM task must have exactly one assigned Groq key; missing={missing}, unknown={unknown}."
    )

# Table-selection fallback may change the model, but the provider keeps the
# table_selection task pinned to key 2. No fallback changes API-key identity.
TASK_FALLBACK_MODELS: dict[str, tuple[str, ...]] = {
    "table_selection": (MODEL_20B,),
}

MAX_ATTEMPTS = 3
RETRY_LIMITS = {
    # A 429 or authentication failure is surfaced immediately. Never switch
    # independent accounts or wait on an application-maintained quota estimate.
    # Transient failures may retry on the same task-assigned key only.
    "rate_limit": 0,
    "authentication": 0,
    "timeout": 1,
    "connection": 1,
    "provider_5xx": 1,
}

RETRY_BACKOFF_SECONDS = 1.0

# Used when a 429 carries no retry hint.
DEFAULT_COOLDOWN_SECONDS = 20.0
MAX_COOLDOWN_SECONDS = 3600.0
# A key that returned 401/403 is taken out of rotation for this long.
AUTH_COOLDOWN_SECONDS = 300.0