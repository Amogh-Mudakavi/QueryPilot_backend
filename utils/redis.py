import redis

from django.conf import settings


_redis_client = None

CHAT_QUESTION_LIMIT = 15
CHAT_COOLDOWN_SECONDS = 2 * 60 * 60  # 2 hours


def get_redis_client():

    global _redis_client

    if _redis_client is None:

        _redis_client = redis.Redis(
            host=getattr(
                settings,
                "REDIS_HOST",
                "127.0.0.1",
            ),
            port=getattr(
                settings,
                "REDIS_PORT",
                6379,
            ),
            db=getattr(
                settings,
                "REDIS_DB",
                0,
            ),
            decode_responses=True,
        )

        _redis_client.ping()

    return _redis_client


def publish_job(job_id):

    client = get_redis_client()

    client.publish(
        "admin_jobs",
        str(job_id),
    )


def get_pubsub():

    client = get_redis_client()

    pubsub = client.pubsub()

    pubsub.subscribe(
        "admin_jobs"
    )

    return pubsub


def check_chat_question_limit(user_id: int) -> tuple[bool, int]:
    """
    Allow up to 10 questions.

    Question 1-10:
        allowed

    Question 11:
        rejected and starts a 2-hour cooldown

    During cooldown:
        rejected without touching the LLM

    Returns:
        (allowed, count)
    """

    client = get_redis_client()

    cooldown_key = (
        f"querypilot:chat:cooldown:{user_id}"
    )

    question_key = (
        f"querypilot:chat:questions:{user_id}"
    )

    # Once cooldown exists, reject immediately.
    if client.exists(cooldown_key):
        count = client.get(question_key)
        return False, int(count or CHAT_QUESTION_LIMIT + 1)

    # INCR is atomic in Redis.
    count = client.incr(question_key)

    # Keep the normal usage window alive for 2 hours.
    if count == 1:
        client.expire(
            question_key,
            CHAT_COOLDOWN_SECONDS,
        )

    # First rejected request is question 11.
    if count > CHAT_QUESTION_LIMIT:

        # Start a fresh 2-hour cooldown from the moment
        # the user actually crosses the limit.
        client.setex(
            cooldown_key,
            CHAT_COOLDOWN_SECONDS,
            "1",
        )

        return False, count

    return True, count