"""Settings module for QueryPilot."""

import os
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


def _env_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _env_list(name, default=None):
    value = os.getenv(name)
    if not value:
        return list(default or [])
    return [item.strip() for item in value.split(",") if item.strip()]


SECRET_KEY = os.getenv("DJANGO_SECRET_KEY")
if not SECRET_KEY:
    raise ImproperlyConfigured(
        "DJANGO_SECRET_KEY is not set. Add it to your .env file."
    )

DEBUG = _env_bool("DJANGO_DEBUG", default=False)

ALLOWED_HOSTS = _env_list(
    "DJANGO_ALLOWED_HOSTS",
    default=["localhost", "127.0.0.1"] if DEBUG else [],
)

if not DEBUG and not ALLOWED_HOSTS:
    raise ImproperlyConfigured(
        "DJANGO_ALLOWED_HOSTS must be configured when DEBUG is False."
    )


INSTALLED_APPS = [
    "corsheaders",
    "rest_framework",
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "accounts",
    "documents",
    "workspace",
    "retrieval",
    "planner",
    "chat",
]


MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"


AUTH_USER_MODEL = "accounts.User"


JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not JWT_SECRET_KEY or len(JWT_SECRET_KEY) < 32:
    raise ImproperlyConfigured(
        "JWT_SECRET_KEY is missing or shorter than 32 characters. "
        "Generate one with: "
        'python -c "import secrets; print(secrets.token_urlsafe(48))" '
        "and put it in your .env file."
    )

JWT_ALGORITHM = "HS256"


GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "").strip()
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "").strip()
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "").strip()

OAUTH_FRONTEND_CALLBACK_URL = os.getenv(
    "OAUTH_FRONTEND_CALLBACK_URL",
    "",
).strip()

OAUTH_STATE_LIFETIME_SECONDS = int(
    os.getenv("OAUTH_STATE_LIFETIME_SECONDS", "600")
)

OAUTH_LOGIN_CODE_LIFETIME_SECONDS = int(
    os.getenv("OAUTH_LOGIN_CODE_LIFETIME_SECONDS", "120")
)

ACCESS_TOKEN_LIFETIME_MINUTES = int(
    os.getenv("ACCESS_TOKEN_LIFETIME_MINUTES", "15")
)

REFRESH_TOKEN_LIFETIME_DAYS = int(
    os.getenv("REFRESH_TOKEN_LIFETIME_DAYS", "7")
)

OTP_LIFETIME_MINUTES = int(
    os.getenv("OTP_LIFETIME_MINUTES", "10")
)

OTP_RESEND_COOLDOWN_SECONDS = int(
    os.getenv("OTP_RESEND_COOLDOWN_SECONDS", "60")
)

OTP_HOURLY_LIMIT = int(
    os.getenv("OTP_HOURLY_LIMIT", "3")
)

OTP_IP_HOURLY_LIMIT = int(
    os.getenv("OTP_IP_HOURLY_LIMIT", "10")
)

OTP_MAX_VERIFICATION_ATTEMPTS = int(
    os.getenv("OTP_MAX_VERIFICATION_ATTEMPTS", "5")
)

LOGIN_RATE_LIMIT_EMAIL_ATTEMPTS = int(
    os.getenv("LOGIN_RATE_LIMIT_EMAIL_ATTEMPTS", "10")
)

LOGIN_RATE_LIMIT_IP_ATTEMPTS = int(
    os.getenv("LOGIN_RATE_LIMIT_IP_ATTEMPTS", "30")
)

LOGIN_RATE_LIMIT_WINDOW_SECONDS = int(
    os.getenv("LOGIN_RATE_LIMIT_WINDOW_SECONDS", "900")
)


if ACCESS_TOKEN_LIFETIME_MINUTES <= 0:
    raise ImproperlyConfigured(
        "ACCESS_TOKEN_LIFETIME_MINUTES must be greater than zero."
    )

if REFRESH_TOKEN_LIFETIME_DAYS <= 0:
    raise ImproperlyConfigured(
        "REFRESH_TOKEN_LIFETIME_DAYS must be greater than zero."
    )

if OTP_LIFETIME_MINUTES <= 0:
    raise ImproperlyConfigured(
        "OTP_LIFETIME_MINUTES must be greater than zero."
    )

if OTP_RESEND_COOLDOWN_SECONDS < 0:
    raise ImproperlyConfigured(
        "OTP_RESEND_COOLDOWN_SECONDS cannot be negative."
    )

if OTP_HOURLY_LIMIT <= 0 or OTP_IP_HOURLY_LIMIT <= 0:
    raise ImproperlyConfigured(
        "OTP hourly limits must be greater than zero."
    )

if OTP_MAX_VERIFICATION_ATTEMPTS <= 0:
    raise ImproperlyConfigured(
        "OTP_MAX_VERIFICATION_ATTEMPTS must be greater than zero."
    )

if (
    LOGIN_RATE_LIMIT_EMAIL_ATTEMPTS <= 0
    or LOGIN_RATE_LIMIT_IP_ATTEMPTS <= 0
    or LOGIN_RATE_LIMIT_WINDOW_SECONDS <= 0
):
    raise ImproperlyConfigured(
        "Login rate-limit settings must be greater than zero."
    )


if DEBUG:
    CORS_ALLOW_ALL_ORIGINS = True
else:
    CORS_ALLOW_ALL_ORIGINS = False

    CORS_ALLOWED_ORIGINS = _env_list(
        "CORS_ALLOWED_ORIGINS",
        default=[],
    )

    CSRF_TRUSTED_ORIGINS = _env_list(
        "CSRF_TRUSTED_ORIGINS",
        default=[],
    )

CORS_EXPOSE_HEADERS = [
    "X-Conversation-Id",
]


SECURE_SSL_REDIRECT = not DEBUG
SESSION_COOKIE_SECURE = not DEBUG
CSRF_COOKIE_SECURE = not DEBUG
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = "DENY"
SECURE_REFERRER_POLICY = "same-origin"

if not DEBUG:
    SECURE_PROXY_SSL_HEADER = (
        "HTTP_X_FORWARDED_PROTO",
        "https",
    )

    SECURE_HSTS_SECONDS = 31536000
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True
else:
    SECURE_HSTS_SECONDS = 0


TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.mysql",
        "NAME": os.getenv(
            "MYSQL_DATABASE",
            "querypilot",
        ),
        "USER": os.getenv(
            "MYSQL_USER",
            "querypilot_admin",
        ),
        "PASSWORD": os.getenv(
            "MYSQL_PASSWORD",
            "",
        ),
        "HOST": os.getenv(
            "MYSQL_HOST",
            "",
        ),
        "PORT": os.getenv(
            "MYSQL_PORT",
            "3306",
        ),
        "OPTIONS": {
            "charset": "utf8mb4",
        },
    },

    "structured": {
        "ENGINE": "django.db.backends.mysql",
        "NAME": os.getenv(
            "STRUCTURED_DB_NAME",
            "querypilot_data",
        ),
        "USER": os.getenv(
            "STRUCTURED_DB_USER",
            "query_structured",
        ),
        "PASSWORD": os.getenv(
            "STRUCTURED_DB_PASSWORD",
            "",
        ),
        "HOST": os.getenv(
            "STRUCTURED_DB_HOST",
            "",
        ),
        "PORT": os.getenv(
            "STRUCTURED_DB_PORT",
            "3306",
        ),
        "OPTIONS": {
            "charset": "utf8mb4",
        },
    },

    "structured_read": {
        "ENGINE": "django.db.backends.mysql",
        "NAME": os.getenv(
            "STRUCTURED_DB_NAME",
            "querypilot_data",
        ),
        "USER": os.getenv(
            "STRUCTURED_DB_READ_USER",
            "",
        ),
        "PASSWORD": os.getenv(
            "STRUCTURED_DB_READ_PASSWORD",
            "",
        ),
        "HOST": os.getenv(
            "STRUCTURED_DB_HOST",
            "",
        ),
        "PORT": os.getenv(
            "STRUCTURED_DB_PORT",
            "3306",
        ),
        "OPTIONS": {
            "charset": "utf8mb4",
        },
    },
}


if not DEBUG and (
    not DATABASES["default"]["USER"]
    or not DATABASES["default"]["PASSWORD"]
    or not DATABASES["default"]["HOST"]
):
    raise ImproperlyConfigured(
        "MYSQL_USER, MYSQL_PASSWORD and MYSQL_HOST "
        "must be configured when DEBUG is False."
    )


if not DEBUG and (
    not DATABASES["structured"]["USER"]
    or not DATABASES["structured"]["PASSWORD"]
    or not DATABASES["structured"]["HOST"]
):
    raise ImproperlyConfigured(
        "STRUCTURED_DB_USER, STRUCTURED_DB_PASSWORD and "
        "STRUCTURED_DB_HOST must be configured when DEBUG is False."
    )

RESEND_API_KEY = os.getenv("RESEND_API_KEY")

HF_TOKEN = os.getenv("HF_TOKEN")

GROQ_API_KEY_1 = os.getenv("GROQ_API_KEY_1")
GROQ_API_KEY_2 = os.getenv("GROQ_API_KEY_2")
GROQ_API_KEY_3 = os.getenv("GROQ_API_KEY_3")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")


AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": (
            "django.contrib.auth.password_validation."
            "UserAttributeSimilarityValidator"
        ),
    },
    {
        "NAME": (
            "django.contrib.auth.password_validation."
            "MinimumLengthValidator"
        ),
    },
    {
        "NAME": (
            "django.contrib.auth.password_validation."
            "CommonPasswordValidator"
        ),
    },
    {
        "NAME": (
            "django.contrib.auth.password_validation."
            "NumericPasswordValidator"
        ),
    },
]


LANGUAGE_CODE = "en-us"
TIME_ZONE = "Asia/Kolkata"
USE_I18N = True
USE_TZ = True


STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"


AWS_STORAGE_BUCKET_NAME = os.getenv("AWS_STORAGE_BUCKET_NAME")
AWS_S3_REGION_NAME = os.getenv(
    "AWS_S3_REGION_NAME",
    "ap-south-1",
)

if not DEBUG and not AWS_STORAGE_BUCKET_NAME:
    raise ImproperlyConfigured(
        "AWS_STORAGE_BUCKET_NAME must be configured when DEBUG is False."
    )

if DEBUG:
    STORAGES = {
        "default": {
            "BACKEND": (
                "django.core.files.storage.FileSystemStorage"
            ),
        },
        "staticfiles": {
            "BACKEND": (
                "django.contrib.staticfiles.storage.StaticFilesStorage"
            ),
        },
    }
else:
    STORAGES = {
        "default": {
            "BACKEND": "storages.backends.s3.S3Storage",
            "OPTIONS": {
                "bucket_name": AWS_STORAGE_BUCKET_NAME,
                "region_name": AWS_S3_REGION_NAME,
                "default_acl": None,
                "querystring_auth": True,
                "querystring_expire": 3600,
                "file_overwrite": False,
            },
        },
        "staticfiles": {
            "BACKEND": (
                "django.contrib.staticfiles.storage.StaticFilesStorage"
            ),
        },
    }


DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"


REST_FRAMEWORK = {
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticated",
    ],
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "accounts.authentication.JWTAuthentication",
    ],
}


REDIS_HOST = os.getenv(
    "REDIS_HOST",
    "localhost",
)

REDIS_PORT = int(
    os.getenv(
        "REDIS_PORT",
        "6379",
    )
)

REDIS_DB = int(
    os.getenv(
        "REDIS_DB",
        "0",
    )
)

REDIS_URL = os.getenv(
    "REDIS_URL",
    f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}",
)

CACHES = {
    "default": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": REDIS_URL,
        "OPTIONS": {
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
        },
    }
}


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "console": {
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "console",
            "stream": "ext://sys.stdout",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": LOG_LEVEL,
    },
}


FILE_UPLOAD_MAX_MEMORY_SIZE = 50 * 1024 * 1024
DATA_UPLOAD_MAX_MEMORY_SIZE = 100 * 1024 * 1024