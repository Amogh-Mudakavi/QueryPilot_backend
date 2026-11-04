"""QueryPilot LLM configuration.

Three things live here:

* the Groq API keys and per-task API-key assignments,
* one ``TaskProfile`` per logical task (model, output cap, timeout, reasoning),
* bounded retry policy for transient provider failures.

There is deliberately NO request-wide token budget, local TPM scheduler, or
API-key rotation. Groq is authoritative for rate-limit enforcement.
``input_hint`` is an observability threshold only: the provider logs a warning
when a prompt exceeds it; it never truncates or refuses the prompt.
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

CHARS_PER_TOKEN_ESTIMATE = 3.5
MESSAGE_OVERHEAD_TOKENS = 4

MODEL_20B = "openai/gpt-oss-20b"
MODEL_120B = "openai/gpt-oss-120b"

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

# Deprecated: retained for compatibility with any remaining imports.
ANSWER_MIN_OUTPUT_TOKENS = 256
ANSWER_TPM_SAFETY_MARGIN_TOKENS = 256
ANSWER_RETRY_MULTIPLIER = 2

# Provider-guided fitting: when Groq rejects a request because input +
# max_tokens exceeds the provider limit, the provider lowers max_tokens by
# exactly the excess Groq reports (plus this margin), but only if what remains
# is still a usable completion for a reasoning model.
MIN_USEFUL_COMPLETION_TOKENS = 1500
FIT_MARGIN_TOKENS = 50

REQUEST_TIMEOUT = 20.0
DEFAULT_TEMPERATURE = 0.0


@dataclass(frozen=True)
class TaskProfile:
    """Generation settings owned by one logical task.

    model            which model runs this task
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

    "scope_guard": TaskProfile(
        MODEL_20B,
        max_tokens=500,
        input_hint=1200,
    ),

    "chitchat": TaskProfile(
        MODEL_20B,
        max_tokens=500,
        reasoning_effort="low",
        input_hint=600,
    ),

    "title": TaskProfile(
        MODEL_20B,
        max_tokens=200,
        reasoning_effort="low",
        input_hint=200,
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

    # Table selection uses GPT-OSS 120B and shares the structured-data lane
    # with SQL generation. Its model fallback stays on this same key.
    "table_selection": TaskProfile(
        MODEL_120B,
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
# Every configured task is pinned to exactly one API key. Model fallback
# retains the same task assignment and never borrows another key.
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
        "The 120B task lanes require GROQ_API_KEY_1, "
        "GROQ_API_KEY_2, and GROQ_API_KEY_3."
    )

if any(
    index < 0 or index >= len(GROQ_API_KEYS)
    for index in TASK_API_KEY_INDEX.values()
):
    raise ValueError("TASK_API_KEY_INDEX references a missing Groq API key.")


TASK_MODELS = {
    name: profile.model
    for name, profile in TASK_PROFILES.items()
}


if set(TASK_API_KEY_INDEX) != set(TASK_PROFILES):
    missing = sorted(set(TASK_PROFILES) - set(TASK_API_KEY_INDEX))
    unknown = sorted(set(TASK_API_KEY_INDEX) - set(TASK_PROFILES))

    raise ValueError(
        f"Every LLM task must have exactly one assigned Groq key; "
        f"missing={missing}, unknown={unknown}."
    )


# Table-selection fallback may change the model, but the provider keeps the
# table_selection task pinned to key 2 (zero-based index 1). No fallback
# changes API-key identity.
TASK_FALLBACK_MODELS: dict[str, tuple[str, ...]] = {
    "table_selection": (MODEL_20B,),
}


MAX_ATTEMPTS = 3

RETRY_LIMITS = {
    # A 429 or authentication failure is surfaced immediately.
    # Transient failures may retry on the same task-assigned key only.
    "rate_limit": 0,
    "authentication": 0,
    "timeout": 1,
    "connection": 1,
    "provider_5xx": 1,
}

RETRY_BACKOFF_SECONDS = 1.0
