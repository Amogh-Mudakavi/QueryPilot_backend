from pathlib import Path
import os

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


# =====================================================
# SECURITY
# =====================================================

SECRET_KEY = os.getenv("DJANGO_SECRET_KEY")
if not SECRET_KEY:
    raise ImproperlyConfigured(
        "DJANGO_SECRET_KEY is not set. Add it to your .env file."
    )

# Fail closed: DEBUG is OFF unless something explicitly turns it on.
# For local dev, add DJANGO_DEBUG=True to your .env file.
DEBUG = _env_bool("DJANGO_DEBUG", default=False)

ALLOWED_HOSTS = _env_list(
    "DJANGO_ALLOWED_HOSTS",
    default=["localhost", "127.0.0.1"] if DEBUG else [],
)

# =====================================================
# APPLICATIONS
# =====================================================

INSTALLED_APPS = [
    # Third Party
    "corsheaders",
    "rest_framework",

    # Django
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",

    # QueryPilot
    "accounts",
    "documents",
    "workspace",
    "retrieval",
    "planner",
    "chat",
]

# =====================================================
# MIDDLEWARE
# =====================================================

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

# =====================================================
# AUTH
# =====================================================

AUTH_USER_MODEL = "accounts.User"

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")

# HS256 needs a properly-sized key (32+ random bytes). A short or
# missing key is a real weakness, not a warning to shrug off — so we
# refuse to start instead of running with a weak key.
if not JWT_SECRET_KEY or len(JWT_SECRET_KEY) < 32:
    raise ImproperlyConfigured(
        "JWT_SECRET_KEY is missing or shorter than 32 characters. "
        "Generate one with: "
        "python -c \"import secrets; print(secrets.token_urlsafe(48))\" "
        "and put it in your .env file."
    )

# Access tokens are short-lived on purpose — refresh tokens are what
# keep the session going. Override via env if you need to.
ACCESS_TOKEN_LIFETIME_MINUTES = int(os.getenv("ACCESS_TOKEN_LIFETIME_MINUTES", "15"))
REFRESH_TOKEN_LIFETIME_DAYS = int(os.getenv("REFRESH_TOKEN_LIFETIME_DAYS", "7"))

# =====================================================
# CORS
# =====================================================

if DEBUG:
    # Convenient for local dev only.
    CORS_ALLOW_ALL_ORIGINS = True
else:
    CORS_ALLOW_ALL_ORIGINS = False
    CORS_ALLOWED_ORIGINS = _env_list("CORS_ALLOWED_ORIGINS", default=[])

CORS_EXPOSE_HEADERS = [
    "X-Conversation-Id",
]

# =====================================================
# TEMPLATES
# =====================================================

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

# =====================================================
# DATABASE
# =====================================================

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.mysql",
        "NAME": "querypilot_backend",
        "USER": "root",
        "PASSWORD": "root",
        "HOST": "127.0.0.1",
        "PORT": "3306",
        "OPTIONS": {
            "charset": "utf8mb4",
        },
    }
}

# =====================================================
# API KEYS / EXTERNAL SERVICES
# =====================================================

RESEND_API_KEY = os.getenv("RESEND_API_KEY")

HF_TOKEN = os.getenv("HF_TOKEN")

# Groq — kept for fallback / later use
GROQ_API_KEY_1 = os.getenv("GROQ_API_KEY_1")
GROQ_API_KEY_2 = os.getenv("GROQ_API_KEY_2")
GROQ_API_KEY_3 = os.getenv("GROQ_API_KEY_3")

# Gemini — active provider for debugging
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

UNSTRUCTURED_API_KEY = os.getenv("UNSTRUCTURED_API_KEY")

# =====================================================
# PASSWORD VALIDATION
# =====================================================

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

# =====================================================
# INTERNATIONALIZATION
# =====================================================

LANGUAGE_CODE = "en-us"

TIME_ZONE = "Asia/Kolkata"

USE_I18N = True

USE_TZ = True

# =====================================================
# STATIC / MEDIA
# =====================================================

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

# =====================================================
# DEFAULT PK
# =====================================================

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# =====================================================
# DRF
# =====================================================

REST_FRAMEWORK = {
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.AllowAny",
    ],

    "DEFAULT_AUTHENTICATION_CLASSES": [
        "accounts.authentication.JWTAuthentication",
    ],
}

# =====================================================
# REDIS
# =====================================================

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

CACHES = {
    "default": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": REDIS_URL,
        "OPTIONS": {
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
        },
    }
}

# =====================================================
# LOGGING
# =====================================================

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
}

# =====================================================
# FILE UPLOADS
# =====================================================

FILE_UPLOAD_MAX_MEMORY_SIZE = 50 * 1024 * 1024

DATA_UPLOAD_MAX_MEMORY_SIZE = 100 * 1024 * 1024