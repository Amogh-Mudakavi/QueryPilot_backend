from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# =====================================================
# SECURITY
# =====================================================

SECRET_KEY = os.getenv("DJANGO_SECRET_KEY")

DEBUG = True

ALLOWED_HOSTS = []

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

# =====================================================
# CORS
# =====================================================

CORS_ALLOW_ALL_ORIGINS = True

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