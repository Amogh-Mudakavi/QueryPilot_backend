import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from django.conf import settings
from llama_cloud import LlamaCloud

from utils.redis import get_redis_client


PARSE_TIMEOUT_SECONDS=600.0
PARSE_POLLING_INTERVAL=2.0
PARSE_MAX_INTERVAL=5.0
PARSE_CACHE_TTL_SECONDS=6*60*60
PARSE_CACHE_MAX_BYTES=15*1024*1024

PARSE_TIER=getattr(
    settings,
    "LLAMA_PARSE_TIER",
    "cost_effective",
)

PARSE_VERSION=getattr(
    settings,
    "LLAMA_PARSE_VERSION",
    "latest",
)

PARSE_CACHE_PREFIX="querypilot:llamaparse:"


class LlamaParseService:
    """
    File -> LlamaParse -> parsed result.

    Parse results are cached in Redis so retries do not need
    to send and parse the same document again.
    """

    client=LlamaCloud(
        api_key=settings.LLAMA_CLOUD_API_KEY,
        timeout=PARSE_TIMEOUT_SECONDS,
    )

    @staticmethod
    def _file_hash(path: Path)->str:

        digest=hashlib.sha256()

        with path.open("rb") as file:

            while True:
                chunk=file.read(1024*1024)

                if not chunk:
                    break

                digest.update(chunk)

        return digest.hexdigest()

    @staticmethod
    def _cache_key(file_hash: str)->str:

        return (
            f"{PARSE_CACHE_PREFIX}"
            f"{PARSE_TIER}:"
            f"{PARSE_VERSION}:"
            f"{file_hash}:items"
        )

    @staticmethod
    def _to_cache_json(result)->str:

        if hasattr(result,"model_dump_json"):

            return result.model_dump_json()

        if hasattr(result,"model_dump"):

            return json.dumps(
                result.model_dump(
                    mode="json",
                )
            )

        if hasattr(result,"dict"):

            return json.dumps(
                result.dict()
            )

        raise TypeError(
            "Unsupported LlamaParse result type."
        )

    @staticmethod
    def _to_namespace(value):

        if isinstance(value,dict):

            return SimpleNamespace(
                **{
                    key:LlamaParseService._to_namespace(item)
                    for key,item in value.items()
                }
            )

        if isinstance(value,list):

            return [
                LlamaParseService._to_namespace(item)
                for item in value
            ]

        return value

    @classmethod
    def _read_cache(cls,cache_key: str):

        client=get_redis_client()

        cached=client.get(cache_key)

        if not cached:
            return None

        try:

            payload=json.loads(cached)

            return cls._to_namespace(payload)

        except Exception:

            client.delete(cache_key)

            return None

    @classmethod
    def _write_cache(
        cls,
        cache_key: str,
        result,
    )->None:

        try:

            payload=cls._to_cache_json(result)

            if len(payload.encode("utf-8"))>PARSE_CACHE_MAX_BYTES:
                return

            get_redis_client().setex(
                cache_key,
                PARSE_CACHE_TTL_SECONDS,
                payload,
            )

        except Exception:

            # Cache failure must never fail ingestion.
            return

    @classmethod
    def parse_document(
        cls,
        file_path: str,
        file_hash: str|None=None,
    ):

        path=Path(file_path)

        if not path.is_file():

            raise FileNotFoundError(
                f"Document file not found at '{file_path}'."
            )

        if file_hash is None:
            file_hash=cls._file_hash(path)

        cache_key=cls._cache_key(file_hash)

        cached_result=cls._read_cache(cache_key)

        if cached_result is not None:
            return cached_result

        uploaded_file=cls.client.files.create(
            file=path,
            purpose="parse",
        )

        result=cls.client.parsing.parse(
            file_id=uploaded_file.id,
            tier=PARSE_TIER,
            version=PARSE_VERSION,
            expand=["items"],
            polling_interval=PARSE_POLLING_INTERVAL,
            max_interval=PARSE_MAX_INTERVAL,
            timeout=PARSE_TIMEOUT_SECONDS,
            backoff="linear",
        )

        cls._write_cache(
            cache_key,
            result,
        )

        return result


class LlamaAdapter:
    """
    Converts LlamaParse output into QueryPilot primitives.

    Output:
        text_nodes
        table_nodes
    """

    @staticmethod
    def extract_text_nodes(result):

        text_nodes=[]

        pages=getattr(
            getattr(result,"items",None),
            "pages",
            [],
        )

        for page in pages:

            page_number=getattr(
                page,
                "page_number",
                None,
            )

            for item in page.items:

                item_type=getattr(
                    item,
                    "type",
                    None,
                )

                if item_type=="table":
                    continue

                # Keep markdown because the chunker uses its headings.

                text=getattr(
                    item,
                    "md",
                    None,
                ) or getattr(
                    item,
                    "value",
                    None,
                )

                if not text:
                    continue

                text_nodes.append(
                    {
                        "page":page_number,
                        "text":text,
                    }
                )

        return text_nodes

    @staticmethod
    def extract_table_nodes(result):

        table_nodes=[]

        pages=getattr(
            getattr(result,"items",None),
            "pages",
            [],
        )

        for page in pages:

            page_number=getattr(
                page,
                "page_number",
                None,
            )

            for item in page.items:

                item_type=getattr(
                    item,
                    "type",
                    None,
                )

                if item_type!="table":
                    continue

                table_nodes.append(
                    {
                        "page":page_number,
                        "rows":getattr(
                            item,
                            "rows",
                            [],
                        ),
                        "csv":getattr(
                            item,
                            "csv",
                            "",
                        ),
                        "html":getattr(
                            item,
                            "html",
                            "",
                        ),
                        "markdown":getattr(
                            item,
                            "md",
                            "",
                        ),
                    }
                )

        return table_nodes
