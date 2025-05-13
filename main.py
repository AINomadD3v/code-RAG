# main.py
import asyncio

# import datetime # Removed - Not used directly
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, List, Optional  # Removed Dict - Not used

import cocoindex
import uvicorn

# --- Attempt to import types from likely submodules ---
# If these paths are incorrect, explore cocoindex package or use Any
try:
    from cocoindex.flows import DataScope, FlowBuilder
    from cocoindex.functions import (
        SentenceTransformerEmbed,  # Built-ins
        SplitRecursively,
    )
    from cocoindex.query import SimpleSemanticsQueryHandler
    from cocoindex.settings import DatabaseConnectionSpec, Settings  # For init
    from cocoindex.sources import LocalFile
    from cocoindex.storages import VectorIndexDef, VectorSimilarityMetric
    from cocoindex.types import DataSlice

    MODULE_IMPORTS_FIXED = True
except ImportError:
    MODULE_IMPORTS_FIXED = False
    # Fallback if specific imports fail (less type safety)
    DataSlice, FlowBuilder, DataScope, VectorIndexDef, VectorSimilarityMetric = (
        Any,
        Any,
        Any,
        Any,
        Any,
    )
    SimpleSemanticsQueryHandler, SentenceTransformerEmbed, SplitRecursively = (
        Any,
        Any,
        Any,
    )
    LocalFile, Settings, DatabaseConnectionSpec = Any, Any, Any
    logger.warning(
        "Could not import specific types from cocoindex submodules. Using Any."
    )
# ------------------------------------------------------

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi import Query as FastAPIQuery
from pydantic import BaseModel

# --- Import your custom operation from the new module ---
from my_ops import extract_extension_op_registered

# --- Basic Logging Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
# ------------------------------------

logger.info(f"--- Imported 'extract_extension_op_registered' from my_ops.py ---")

# --- Global State & Configuration ---
_COCOINDEX_INITIALIZED = False
_FLOW_DEFINED_AND_REGISTERED = False

_FLOW_OBJECT: Optional[cocoindex.flow.Flow] = (
    None  # Assuming cocoindex.flow.Flow is correct top-level type
)
_QUERY_HANDLER: Optional[SimpleSemanticsQueryHandler] = None  # Use imported type

_CODE_EMBEDDING_FLOW_NAME = "CodeEmbedding"

load_dotenv()
logger.info(f"--- {__file__}: Module loading/reloading ---")


# --- 1. CocoIndex Initialization ---
def initialize_cocoindex():
    global _COCOINDEX_INITIALIZED
    if _COCOINDEX_INITIALIZED:
        return
    logger.info("--- Initializing CocoIndex library ---")
    db_url = os.getenv("COCOINDEX_DATABASE_URL")
    if not db_url:
        logger.critical("COCOINDEX_DATABASE_URL environment variable not set!")
        raise ValueError("COCOINDEX_DATABASE_URL must be set.")
    try:
        # Use imported Settings types if available
        cocoindex.init(Settings(database=DatabaseConnectionSpec(url=db_url)))
        _COCOINDEX_INITIALIZED = True
        logger.info("--- CocoIndex library initialized successfully ---")
    except Exception as e:
        logger.critical(f"cocoindex.init() failed: {e}", exc_info=True)
        raise RuntimeError(f"Failed to initialize CocoIndex: {e}")


initialize_cocoindex()


# --- 2. Define Other Operations ---
def _code_to_embedding_logic_impl(text: DataSlice) -> DataSlice:  # Use imported type
    # Use imported SentenceTransformerEmbed if available
    return text.transform(
        SentenceTransformerEmbed(model="sentence-transformers/all-MiniLM-L6-v2")
    )


code_to_embedding_logic_op: Any = _code_to_embedding_logic_impl


# --- 3. Define Flow Builder Function (WITHOUT decorator) ---
def code_embedding_flow_builder_func(
    flow_builder: FlowBuilder, data_scope: DataScope  # Use imported types
):
    logger.info(
        f"Defining flow logic structure for '{_CODE_EMBEDDING_FLOW_NAME}' (registration in lifespan)"
    )
    # Use imported LocalFile if available
    data_scope["files"] = flow_builder.add_source(
        LocalFile(
            path="uiautomator2",
            included_patterns=["*.py"],
            excluded_patterns=[".*", "**/__pycache__/**", "**/tests/**"],
        )
    )
    code_embeddings_collector = data_scope.add_collector()
    with data_scope["files"].row() as file:
        # --- FIX: Use .call() for custom registered operations ---
        file["extension"] = file["filename"].call(extract_extension_op_registered)
        # -------------------------------------------------------
        # Use imported SplitRecursively if available
        file["chunks"] = file["content"].transform(
            SplitRecursively(),
            language=file["extension"],
            chunk_size=300,
            chunk_overlap=50,
        )
        with file["chunks"].row() as chunk:
            chunk["embedding"] = chunk["text"].call(code_to_embedding_logic_op)
            code_embeddings_collector.collect(
                filename=file["filename"],
                location=chunk["location"],
                text=chunk["text"],
                embedding=chunk["embedding"],
            )
    code_embeddings_collector.export(
        "code_embeddings",
        cocoindex.storages.Postgres(),  # Assuming this path is correct
        primary_key_fields=["filename", "location"],
        vector_indexes=[
            # Use imported VectorIndexDef and VectorSimilarityMetric if available
            VectorIndexDef(
                field_name="embedding",
                metric=VectorSimilarityMetric.COSINE_SIMILARITY,
            )
        ],
    )


# --- 4. Query Handler Setup (Lazy Initialization) ---
def get_query_handler() -> SimpleSemanticsQueryHandler:  # Use imported type
    global _QUERY_HANDLER, _FLOW_OBJECT
    if _QUERY_HANDLER:
        return _QUERY_HANDLER

    if not _COCOINDEX_INITIALIZED:
        logger.warning(
            "Query Handler Init: CocoIndex not initialized. Attempting init."
        )
        initialize_cocoindex()

    if _FLOW_OBJECT is None:
        logger.critical(
            f"Query Handler Init: Flow object '{_CODE_EMBEDDING_FLOW_NAME}' is None after lifespan startup."
        )
        raise RuntimeError(
            f"Flow object '{_CODE_EMBEDDING_FLOW_NAME}' was not initialized correctly."
        )

    logger.info(
        f"Initializing Query Handler 'CodeSearch' using Flow '{_FLOW_OBJECT.name if hasattr(_FLOW_OBJECT, 'name') else _CODE_EMBEDDING_FLOW_NAME}'..."
    )

    def query_text_to_embedding(text: DataSlice) -> DataSlice:  # Use imported type
        # Use imported SentenceTransformerEmbed if available
        return text.transform(
            SentenceTransformerEmbed(model="sentence-transformers/all-MiniLM-L6-v2")
        )

    _QUERY_HANDLER = SimpleSemanticsQueryHandler(
        name="CodeSearch",
        flow=_FLOW_OBJECT,
        target_name="code_embeddings",
        query_transform_flow=query_text_to_embedding,
        # Use imported VectorSimilarityMetric if available
        default_similarity_metric=VectorSimilarityMetric.COSINE_SIMILARITY,
    )
    logger.info("Query handler 'CodeSearch' initialized successfully.")
    return _QUERY_HANDLER


# --- 5. FastAPI Application Definition ---
app = FastAPI(
    title="CocoIndex uiautomator2 Code Search API",
    description="API to perform semantic search on the uiautomator2 codebase.",
    version="0.1.0",
)


# Lifespan context manager
@asynccontextmanager
async def lifespan(
    current_app: FastAPI,
):  # Can ignore unused 'current_app' hint or use _
    global _FLOW_OBJECT, _FLOW_DEFINED_AND_REGISTERED, _QUERY_HANDLER

    logger.info("--- FastAPI lifespan startup ---")
    if not _COCOINDEX_INITIALIZED:
        initialize_cocoindex()

    if not _FLOW_DEFINED_AND_REGISTERED:
        logger.info(
            f"Lifespan: Defining and registering flow '{_CODE_EMBEDDING_FLOW_NAME}'..."
        )
        try:
            flow_obj = cocoindex.flow.add_flow_def(
                _CODE_EMBEDDING_FLOW_NAME, code_embedding_flow_builder_func
            )
            _FLOW_OBJECT = flow_obj
            _FLOW_DEFINED_AND_REGISTERED = True
            logger.info(
                f"Lifespan: Flow '{_CODE_EMBEDDING_FLOW_NAME}' registered successfully."
            )
        except KeyError:
            logger.warning(
                f"Lifespan: Flow '{_CODE_EMBEDDING_FLOW_NAME}' already existed. Retrieving."
            )
            try:
                # Pyright doesn't see get_flow_by_name, but keep runtime check
                if hasattr(cocoindex.flow, "get_flow_by_name"):
                    _FLOW_OBJECT = cocoindex.flow.get_flow_by_name(_CODE_EMBEDDING_FLOW_NAME)  # type: ignore
                    _FLOW_DEFINED_AND_REGISTERED = True
                else:
                    raise RuntimeError(
                        "Flow already exists but get_flow_by_name unavailable."
                    )
            except Exception as e:
                logger.critical(f"Lifespan flow retrieval failed: {e}", exc_info=True)
                raise
        except Exception as e:
            logger.critical(f"Lifespan flow definition failed: {e}", exc_info=True)
            raise
    else:
        logger.info(
            f"Lifespan: Flow '{_CODE_EMBEDDING_FLOW_NAME}' assumed already registered."
        )
        if _FLOW_OBJECT is None:
            logger.warning(
                "Lifespan: Flow flag true but object is None. Attempting retrieval."
            )
            try:
                if hasattr(cocoindex.flow, "get_flow_by_name"):
                    _FLOW_OBJECT = cocoindex.flow.get_flow_by_name(_CODE_EMBEDDING_FLOW_NAME)  # type: ignore
                if _FLOW_OBJECT is None:
                    raise RuntimeError(
                        "Flow flag true, object None, retrieval failed/unavailable."
                    )
            except Exception as e:
                logger.critical(
                    f"Lifespan flow retrieval (flag true) failed: {e}", exc_info=True
                )
                raise

    logger.info("Lifespan: Initializing query handler...")
    try:
        get_query_handler()
        logger.info("Lifespan: Query Handler initialization complete.")
    except Exception as e:
        logger.critical(
            f"Lifespan Query Handler initialization failed: {e}", exc_info=True
        )

    logger.info("--- FastAPI lifespan startup complete. Server ready. ---")
    yield
    logger.info("--- FastAPI lifespan shutdown. ---")


app.router.lifespan_context = lifespan


# --- API Models ---
class SearchResultItem(BaseModel):
    filename: Optional[str] = None
    location: Optional[str] = None
    text: Optional[str] = None
    score: float


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResultItem]
    message: Optional[str] = None


# --- API Endpoints ---
@app.get("/search", response_model=SearchResponse)
async def api_search_codebase(
    query: str = FastAPIQuery(
        ..., min_length=1, description="The search query string."
    ),
    top_k: int = FastAPIQuery(
        5, ge=1, le=20, description="Number of results to return."
    ),
):
    try:
        query_handler = get_query_handler()
    except Exception as e:
        logger.error(f"API Error: Failed to get Query Handler: {e}", exc_info=True)
        raise HTTPException(
            status_code=503, detail=f"Search service not ready: {str(e)}"
        )

    if not query_handler:
        raise HTTPException(status_code=503, detail="Search service is not ready.")

    logger.info(f"Received search request: query='{query}', top_k={top_k}")

    try:
        # Log the type, not a potentially non-existent '.name'
        logger.info(
            f"Performing search using handler type '{type(query_handler).__name__}'..."
        )

        # --- FIX: Use 'limit' instead of 'top_k' ---
        search_result_items, _ = query_handler.search(query, limit=top_k)
        # -------------------------------------------

        logger.info(f"Raw result count from handler: {len(search_result_items)}")
        logger.debug(
            f"Raw results received from query_handler.search: {search_result_items}"
        )

    except TypeError as e:
        # Catch if 'limit' is also wrong or other arguments are missing/unexpected
        logger.error(
            f"API Error: TypeError during search call (check parameters like 'limit'): {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Search handler parameter error: {str(e)}"
        )
    except AttributeError as e:
        logger.error(
            f"API Error: Query handler missing 'search' method or other attribute: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Search handler configuration/usage error: {str(e)}",
        )
    except Exception as e:
        logger.error(f"API Error during search execution: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error performing search: {str(e)}"
        )

    processed_results = []
    if search_result_items:
        logger.info(f"Processing {len(search_result_items)} search results.")
        for i, r in enumerate(search_result_items):
            logger.debug(
                f"Result {i+1} - Score: {getattr(r, 'score', 'N/A'):.4f}, Data: {getattr(r, 'data', {})}"
            )
            processed_results.append(
                SearchResultItem(
                    filename=r.data.get("filename"),
                    location=str(r.data.get("location", "")),
                    text=r.data.get("text"),
                    score=getattr(r, "score", 0.0),
                )
            )
    else:
        logger.info("Search returned no results.")

    return SearchResponse(
        query=query, results=processed_results, message="Search successful"
    )


@app.get("/health")
async def health_check():
    handler_ready = _QUERY_HANDLER is not None
    flow_ready = _FLOW_OBJECT is not None
    handler_init_success = False
    if handler_ready:
        try:
            # Check a known method/attribute exists instead of '.name'
            handler_init_success = hasattr(_QUERY_HANDLER, "search")
        except Exception:
            handler_init_success = False
    status = "degraded"
    if _COCOINDEX_INITIALIZED and flow_ready and handler_init_success:
        status = "ok"
    logger.debug(f"Health check status: {status}")
    return {
        "status": status,
        "cocoindex_initialized": _COCOINDEX_INITIALIZED,
        "flow_object_available": flow_ready,
        "query_handler_ready": handler_init_success,
    }


# --- 6. CocoIndex CLI Integration & API Server Runner ---
@cocoindex.main_fn()
async def _cocoindex_cli_entrypoint():
    is_cocoinsight_server_command = (
        len(sys.argv) > 1
        and sys.argv[1] == "cocoindex"
        and len(sys.argv) > 2
        and sys.argv[2] == "server"
    )
    if is_cocoinsight_server_command:
        logger.info(
            "CocoIndex 'server' (for CocoInsight) detected. Keeping process alive..."
        )
        await asyncio.Event().wait()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "serve-api":
        logger.info("Command: 'serve-api'. Starting FastAPI/Uvicorn server...")
        uvicorn.run(
            "main:app",
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            reload=False,  # Keep reload=False
            log_level="info",
        )
    elif len(sys.argv) > 1 and sys.argv[1] == "cocoindex":
        logger.info(
            f"Command: 'cocoindex'. Handing over to CocoIndex CLI for: '{' '.join(sys.argv[2:])}'"
        )
        asyncio.run(_cocoindex_cli_entrypoint())
    else:
        print(
            "--- No specific command. Use 'serve-api' or 'cocoindex <subcommand>'. ---"
        )
        # ... (usage examples unchanged) ...
    logger.info(f"--- Script {__file__} __main__ block finished. ---")
