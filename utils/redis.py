import redis

from django.conf import settings


_redis_client = None


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