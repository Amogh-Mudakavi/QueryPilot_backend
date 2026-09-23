import threading

import redis
from django.conf import settings


_redis_client=None
_redis_lock=threading.Lock()

INGESTION_QUEUE="querypilot:ingestion:queue"

REDIS_CONNECT_TIMEOUT=5
REDIS_SOCKET_TIMEOUT=15
INGESTION_POP_TIMEOUT=5

CHAT_QUESTION_LIMIT=50
CHAT_COOLDOWN_SECONDS=2*60*60


def get_redis_client():

    global _redis_client

    if _redis_client is None:

        with _redis_lock:

            if _redis_client is None:

                _redis_client=redis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    db=settings.REDIS_DB,
                    socket_connect_timeout=REDIS_CONNECT_TIMEOUT,
                    socket_timeout=REDIS_SOCKET_TIMEOUT,
                    socket_keepalive=True,
                    health_check_interval=30,
                    decode_responses=True,
                )

    return _redis_client


def enqueue_job(job_id):

    client=get_redis_client()

    client.rpush(
        INGESTION_QUEUE,
        str(job_id),
    )


def publish_job(job_id):

    # Kept for compatibility with the existing upload API.
    enqueue_job(job_id)


def get_next_job(
    timeout=INGESTION_POP_TIMEOUT,
):

    client=get_redis_client()

    try:

        result=client.blpop(
            INGESTION_QUEUE,
            timeout=timeout,
        )

    except redis.exceptions.TimeoutError:

        # A normal idle period must not kill the worker.
        return None

    if result is None:
        return None

    _,job_id=result

    try:
        return int(job_id)

    except (
        TypeError,
        ValueError,
    ):
        return None


def check_chat_question_limit(user_id: int) -> tuple[bool, int]:
    """
    Allow up to 50 questions.

    Question 1-50:
        allowed

    Question 51:
        rejected and starts a 2-hour cooldown

    During cooldown:
        rejected without touching the LLM

    Returns:
        (allowed, count)
    """

    client=get_redis_client()

    cooldown_key=(
        f"querypilot:chat:cooldown:{user_id}"
    )

    question_key=(
        f"querypilot:chat:questions:{user_id}"
    )

    if client.exists(cooldown_key):

        count=client.get(
            question_key
        )

        return (
            False,
            int(
                count
                or CHAT_QUESTION_LIMIT+1
            ),
        )

    count=client.incr(
        question_key
    )

    if count==1:

        client.expire(
            question_key,
            CHAT_COOLDOWN_SECONDS,
        )

    if count>CHAT_QUESTION_LIMIT:

        client.setex(
            cooldown_key,
            CHAT_COOLDOWN_SECONDS,
            "1",
        )

        return (
            False,
            count,
        )

    return (
        True,
        count,
    )