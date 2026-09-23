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

DEBUG = _env_bool("DJANGO_DEBUG", default=False)

ALLOWED_HOSTS = _env_list(
    "DJANGO_ALLOWED_HOSTS",
    default=["localhost", "127.0.0.1"] if DEBUG else [],
)

if not DEBUG and not ALLOWED_HOSTS:
    raise ImproperlyConfigured(
        "DJANGO_ALLOWED_HOSTS must be configured when DEBUG is False."
    )

# =====================================================
# APPLICATIONS
# =====================================================

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

# JWT configuration has one owner: settings.py.
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not JWT_SECRET_KEY or len(JWT_SECRET_KEY) < 32:
    raise ImproperlyConfigured(
        "JWT_SECRET_KEY is missing or shorter than 32 characters. "
        "Generate one with: "
        "python -c \"import secrets; print(secrets.token_urlsafe(48))\" "
        "and put it in your .env file."
    )

JWT_ALGORITHM = "HS256"

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

# =====================================================
# CORS / HTTPS
# =====================================================

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
    # Expected when Django is behind an HTTPS reverse proxy.
    SECURE_PROXY_SSL_HEADER = (
        "HTTP_X_FORWARDED_PROTO",
        "https",
    )

    SECURE_HSTS_SECONDS = 31536000
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True
else:
    SECURE_HSTS_SECONDS = 0

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
        "NAME": os.getenv(
            "MYSQL_DATABASE",
            "querypilot_backend",
        ),
        "USER": os.getenv(
            "MYSQL_USER",
            "root" if DEBUG else "",
        ),
        "PASSWORD": os.getenv(
            "MYSQL_PASSWORD",
            "root" if DEBUG else "",
        ),
        "HOST": os.getenv(
            "MYSQL_HOST",
            "127.0.0.1",
        ),
        "PORT": os.getenv(
            "MYSQL_PORT",
            "3306",
        ),
        "OPTIONS": {
            "charset": "utf8mb4",
        },
    }
}

if not DEBUG and (
    not DATABASES["default"]["USER"]
    or not DATABASES["default"]["PASSWORD"]
):
    raise ImproperlyConfigured(
        "MYSQL_USER and MYSQL_PASSWORD must be configured when DEBUG is False."
    )

# =====================================================
# API KEYS / EXTERNAL SERVICES
# =====================================================

RESEND_API_KEY = os.getenv("RESEND_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

GROQ_API_KEY_1 = os.getenv("GROQ_API_KEY_1")
GROQ_API_KEY_2 = os.getenv("GROQ_API_KEY_2")
GROQ_API_KEY_3 = os.getenv("GROQ_API_KEY_3")

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
    # Protected by default. Public auth endpoints opt in with AllowAny.
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticated",
    ],
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "accounts.authentication.JWTAuthentication",
    ],
}

# =====================================================
# REDIS
# =====================================================

REDIS_HOST = os.getenv(
    "REDIS_HOST",
    "localhost",
)

REDIS_PORT = int(
    os.getenv("REDIS_PORT", "6379")
)

REDIS_DB = int(
    os.getenv("REDIS_DB", "0")
)

REDIS_URL = (
    f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
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