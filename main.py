# RagChunking/main.py

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, List, Optional  # Kept for general type hinting

import cocoindex  # Main import first
import uvicorn

# --- Basic Logging Configuration ---
# Placed at the top to be available for all subsequent code
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)  # Use __name__ for module-specific logger
# ------------------------------------

# --- Attempt to import specific components if they are public submodules ---
# These are often correct as CocoIndex needs to expose them for its own APIs
try:
    from cocoindex.functions import SentenceTransformerEmbed, SplitRecursively
    from cocoindex.query import SimpleSemanticsQueryHandler
    from cocoindex.sources import LocalFile

    # Settings, DatabaseConnectionSpec, DataSlice, FlowBuilder, DataScope,
    # VectorIndexDef, VectorSimilarityMetric will be accessed as cocoindex.TypeName
    MODULE_IMPORTS_SUCCESSFUL = True
except ImportError as e:
    logger.error(
        f"Failed to import specific CocoIndex components (query, functions, sources): {e}. "
        "App may fall back to Any types or fail if these are critical.",
        exc_info=True,
    )
    MODULE_IMPORTS_SUCCESSFUL = False
    # Define fallbacks if necessary, though the app might not function
    SimpleSemanticsQueryHandler = Any
    SentenceTransformerEmbed, SplitRecursively = Any, Any
    LocalFile = Any
# --------------------------------------------------------------------------

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi import Query as FastAPIQuery
from fastapi.middleware.cors import CORSMiddleware  # Import CORS
from pydantic import BaseModel

from my_ops import extract_extension_op_registered  # Your custom ops module

logger.info(f"--- Imported 'extract_extension_op_registered' from my_ops.py ---")

# --- Global State & Configuration ---
_COCOINDEX_INITIALIZED = False
_FLOW_DEFINED_AND_REGISTERED = False

_FLOW_OBJECT: Optional[cocoindex.flow.Flow] = None  # Main type for a flow object
_QUERY_HANDLER: Optional[SimpleSemanticsQueryHandler] = (
    None  # Type from specific import
)

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
        # Use cocoindex.Settings and cocoindex.DatabaseConnectionSpec directly
        init_settings = cocoindex.Settings(
            database=cocoindex.DatabaseConnectionSpec(url=db_url)
        )
        cocoindex.init(init_settings)
        _COCOINDEX_INITIALIZED = True
        logger.info("--- CocoIndex library initialized successfully ---")
    except AttributeError as ae:
        logger.critical(
            f"cocoindex.init() failed due to missing attributes "
            f"(Settings/DatabaseConnectionSpec not on 'cocoindex' module?): {ae}",
            exc_info=True,
        )
        raise RuntimeError(
            f"Failed to initialize CocoIndex due to AttributeError: {ae}"
        )
    except Exception as e:
        logger.critical(f"cocoindex.init() failed: {e}", exc_info=True)
        raise RuntimeError(f"Failed to initialize CocoIndex: {e}")


initialize_cocoindex()


# --- 2. Define Other Operations ---
# Using cocoindex.DataSlice for type hints as suggested by docs for similar types
def _code_to_embedding_logic_impl(text: cocoindex.DataSlice) -> cocoindex.DataSlice:
    return text.transform(
        SentenceTransformerEmbed(  # This comes from the specific import
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
    )


code_to_embedding_logic_op: Any = _code_to_embedding_logic_impl


# --- 3. Define Flow Builder Function (WITHOUT decorator) ---
# Using cocoindex.FlowBuilder and cocoindex.DataScope as per documentation
def code_embedding_flow_builder_func(
    flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope
):
    logger.info(
        f"Defining flow logic structure for '{_CODE_EMBEDDING_FLOW_NAME}' (registration in lifespan)"
    )
    data_scope["files"] = flow_builder.add_source(
        LocalFile(  # This comes from the specific import
            path="uiautomator2",  # Make sure this path exists relative to execution
            included_patterns=["*.py"],
            excluded_patterns=[".*", "**/__pycache__/**", "**/tests/**"],
        )
    )
    code_embeddings_collector = data_scope.add_collector()
    with data_scope["files"].row() as file:
        # Use .transform() for the custom operation handle from @cocoindex.op.function
        file["extension"] = file["filename"].transform(extract_extension_op_registered)
        file["chunks"] = file["content"].transform(
            SplitRecursively(),  # This comes from the specific import
            language=file["extension"],
            chunk_size=300,
            chunk_overlap=50,
        )
        with file["chunks"].row() as chunk:
            # Use .call() for plain Python functions
            chunk["embedding"] = chunk["text"].call(code_to_embedding_logic_op)
            code_embeddings_collector.collect(
                filename=file["filename"],
                location=chunk["location"],
                text=chunk["text"],
                embedding=chunk["embedding"],
            )
    code_embeddings_collector.export(
        "code_embeddings",
        cocoindex.storages.Postgres(),  # Assuming cocoindex.storages.Postgres is correct
        primary_key_fields=["filename", "location"],
        vector_indexes=[
            # Using cocoindex.VectorIndexDef and cocoindex.VectorSimilarityMetric as per documentation
            cocoindex.VectorIndexDef(
                field_name="embedding",
                metric=cocoindex.VectorSimilarityMetric.COSINE_SIMILARITY,
            )
        ],
    )


# --- 4. Query Handler Setup (Lazy Initialization) ---
def get_query_handler() -> SimpleSemanticsQueryHandler:  # Type from specific import
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

    flow_name_for_log = _CODE_EMBEDDING_FLOW_NAME
    # Defensively check for 'name' attribute on the flow object
    if (
        hasattr(_FLOW_OBJECT, "name")
        and isinstance(_FLOW_OBJECT.name, str)
        and _FLOW_OBJECT.name
    ):
        flow_name_for_log = _FLOW_OBJECT.name
    elif hasattr(_FLOW_OBJECT, "name") and callable(
        _FLOW_OBJECT.name
    ):  # If .name is a method
        try:
            name_val = _FLOW_OBJECT.name()
            if isinstance(name_val, str) and name_val:
                flow_name_for_log = name_val
        except Exception:
            logger.warning(
                "Could not retrieve flow name by calling _FLOW_OBJECT.name()"
            )

    logger.info(
        f"Initializing Query Handler 'CodeSearch' using Flow '{flow_name_for_log}'..."
    )

    # Using cocoindex.DataSlice for type hints
    def query_text_to_embedding(text: cocoindex.DataSlice) -> cocoindex.DataSlice:
        return text.transform(
            SentenceTransformerEmbed(  # From specific import
                model="sentence-transformers/all-MiniLM-L6-v2"
            )
        )

    _QUERY_HANDLER = SimpleSemanticsQueryHandler(
        name="CodeSearch",
        flow=_FLOW_OBJECT,
        target_name="code_embeddings",
        query_transform_flow=query_text_to_embedding,
        # Using cocoindex.VectorSimilarityMetric as per documentation
        default_similarity_metric=cocoindex.VectorSimilarityMetric.COSINE_SIMILARITY,
    )
    logger.info("Query handler 'CodeSearch' initialized successfully.")
    return _QUERY_HANDLER


# --- 5. FastAPI Application Definition ---
# This 'app' is for the RAG API server (RagChunking/main.py)
app = FastAPI(
    title="RAG API Server for Code Search",
    description="API to perform semantic search on a codebase.",
    version="0.1.0",
    # lifespan context will be assigned after CORS middleware
)

# --- ADD CORS MIDDLEWARE ---
# Define the origins that are allowed to make requests.
# This should be the origin of your uiautodev frontend.
origins = [
    "http://localhost:20242",  # Default port for uiautodev if run locally
    "http://127.0.0.1:20242",  # Common alternative for localhost
    # Add other origins if your frontend might be served from elsewhere during development/testing
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)
# --------------------------


@asynccontextmanager
async def lifespan(current_app: FastAPI):  # current_app can be changed to _ if not used
    global _FLOW_OBJECT, _FLOW_DEFINED_AND_REGISTERED, _QUERY_HANDLER  # Ensure _QUERY_HANDLER is global if modified

    logger.info("--- RAG API FastAPI lifespan startup ---")
    if not _COCOINDEX_INITIALIZED:  # Should be done by top-level call, but ensure
        initialize_cocoindex()

    if not _FLOW_DEFINED_AND_REGISTERED:
        logger.info(
            f"RAG Lifespan: Defining and registering flow '{_CODE_EMBEDDING_FLOW_NAME}'..."
        )
        try:
            # Assuming cocoindex.flow.add_flow_def exists as per docs
            flow_obj = cocoindex.flow.add_flow_def(
                _CODE_EMBEDDING_FLOW_NAME, code_embedding_flow_builder_func
            )
            _FLOW_OBJECT = flow_obj
            _FLOW_DEFINED_AND_REGISTERED = True
            logger.info(
                f"RAG Lifespan: Flow '{_CODE_EMBEDDING_FLOW_NAME}' registered successfully."
            )
        except KeyError:  # If add_flow_def raises KeyError on duplicate
            logger.warning(
                f"RAG Lifespan: Flow '{_CODE_EMBEDDING_FLOW_NAME}' already existed. Retrieving."
            )
            try:
                if hasattr(cocoindex.flow, "get_flow_by_name"):
                    _FLOW_OBJECT = cocoindex.flow.get_flow_by_name(_CODE_EMBEDDING_FLOW_NAME)  # type: ignore
                    _FLOW_DEFINED_AND_REGISTERED = True  # Mark as handled
                else:
                    raise RuntimeError(
                        "Flow already exists but get_flow_by_name unavailable."
                    )
            except Exception as e_retrieve:
                logger.critical(
                    f"RAG Lifespan flow retrieval failed: {e_retrieve}", exc_info=True
                )
                raise
        except Exception as e_define:
            logger.critical(
                f"RAG Lifespan flow definition failed: {e_define}", exc_info=True
            )
            raise
    else:
        logger.info(
            f"RAG Lifespan: Flow '{_CODE_EMBEDDING_FLOW_NAME}' assumed already registered."
        )
        if (
            _FLOW_OBJECT is None
        ):  # If flag was true but object is None (e.g. Uvicorn worker weirdness)
            logger.warning(
                "RAG Lifespan: Flow flag true but object is None. Attempting retrieval."
            )
            try:
                if hasattr(cocoindex.flow, "get_flow_by_name"):
                    _FLOW_OBJECT = cocoindex.flow.get_flow_by_name(_CODE_EMBEDDING_FLOW_NAME)  # type: ignore
                if _FLOW_OBJECT is None:  # After attempt
                    raise RuntimeError(
                        "Flow flag true, object None, retrieval failed/unavailable."
                    )
            except Exception as e_retrieve_flagged:
                logger.critical(
                    f"RAG Lifespan flow retrieval (flag true) failed: {e_retrieve_flagged}",
                    exc_info=True,
                )
                raise

    logger.info("RAG Lifespan: Initializing query handler...")
    try:
        get_query_handler()  # This will initialize _QUERY_HANDLER
        logger.info("RAG Lifespan: Query Handler initialization complete.")
    except Exception as e_qh_init:
        logger.critical(
            f"RAG Lifespan Query Handler initialization failed: {e_qh_init}",
            exc_info=True,
        )
        # Depending on requirements, you might want to raise here to stop startup if handler is critical
        # raise

    logger.info("--- RAG API FastAPI lifespan startup complete. Server ready. ---")
    yield
    logger.info("--- RAG API FastAPI lifespan shutdown. ---")


# Assign the lifespan to the app *after* defining it and the app instance
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

    if not query_handler:  # Should be caught by the exception above, but as a safeguard
        raise HTTPException(
            status_code=503, detail="Search service is not ready (handler is None)."
        )

    logger.info(
        f"RAG API: Received search request: query='{query}', top_k (limit will be)={top_k}"
    )

    try:
        handler_type_name = type(query_handler).__name__
        logger.info(
            f"RAG API: Performing search using handler type '{handler_type_name}'..."
        )
        # Use 'limit' as the keyword argument for the search method
        search_result_items, _ = query_handler.search(query, limit=top_k)

        logger.info(
            f"RAG API: Raw result count from handler: {len(search_result_items)}"
        )
        # Using DEBUG level for potentially verbose raw results
        logger.debug(
            f"RAG API: Raw results received from query_handler.search: {search_result_items}"
        )

    except TypeError as e:
        # This will catch if 'limit' is still wrong or other arguments are missing/unexpected
        logger.error(
            f"RAG API Error: TypeError during search call (check parameters like 'limit'): {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Search handler parameter error: {str(e)}"
        )
    except AttributeError as e:
        # This could be if query_handler itself is None or if 'search' method is genuinely missing
        logger.error(
            f"RAG API Error: Query handler missing 'search' method or other attribute: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Search handler configuration/usage error: {str(e)}",
        )
    except Exception as e:
        logger.error(f"RAG API Error during search execution: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error performing search: {str(e)}"
        )

    processed_results = []
    if search_result_items:
        logger.info(f"RAG API: Processing {len(search_result_items)} search results.")
        for i, r in enumerate(search_result_items):
            # Safe attribute access for score and data, providing defaults
            score = getattr(r, "score", 0.0)
            data_dict = getattr(r, "data", {})
            logger.debug(
                f"RAG API Result {i+1} - Score: {score:.4f}, Data: {data_dict}"
            )
            processed_results.append(
                SearchResultItem(
                    filename=data_dict.get("filename"),
                    location=str(
                        data_dict.get("location", "")
                    ),  # Ensure location is string
                    text=data_dict.get("text"),
                    score=score,
                )
            )
    else:
        logger.info("RAG API: Search returned no results.")

    return SearchResponse(
        query=query, results=processed_results, message="Search successful"
    )


@app.get("/health")
async def health_check():
    handler_ready = _QUERY_HANDLER is not None
    flow_ready = _FLOW_OBJECT is not None
    handler_init_success = False  # Default to false

    if handler_ready:
        try:
            # Check for a known method rather than a potentially non-existent '.name'
            handler_init_success = hasattr(_QUERY_HANDLER, "search")
        except Exception:
            # In case hasattr itself fails on a malformed _QUERY_HANDLER or proxy
            handler_init_success = False

    # Log the values being checked for health status
    logger.info(
        f"RAG API Health Check Details: cocoindex_initialized={_COCOINDEX_INITIALIZED}, "
        f"flow_ready={flow_ready}, handler_init_success={handler_init_success}"
    )

    current_status = "degraded"
    if _COCOINDEX_INITIALIZED and flow_ready and handler_init_success:
        current_status = "ok"

    return {
        "status": current_status,
        "cocoindex_initialized": _COCOINDEX_INITIALIZED,
        "flow_object_available": flow_ready,
        "query_handler_ready": handler_init_success,
    }


# --- 6. CocoIndex CLI Integration & API Server Runner ---
@cocoindex.main_fn()
async def _cocoindex_cli_entrypoint():
    # This function's body is mainly for keeping 'cocoindex server' (CocoInsight) alive.
    # Other CLI commands are handled by CocoIndex before this body is typically run.
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
        logger.info("Command: 'serve-api'. Starting RAG API FastAPI/Uvicorn server...")
        uvicorn.run(
            "main:app",  # Points to the 'app' FastAPI instance in this 'main.py' file
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            reload=False,  # IMPORTANT: Keep reload=False to prevent re-registration issues
            log_level="info",  # Uvicorn's log level
        )
    elif len(sys.argv) > 1 and sys.argv[1] == "cocoindex":
        logger.info(
            f"Command: 'cocoindex'. Handing over to CocoIndex CLI for: '{' '.join(sys.argv[2:])}'"
        )
        asyncio.run(_cocoindex_cli_entrypoint())
    else:
        print(
            "--- No specific command for RAG API. Use 'serve-api' or 'cocoindex <subcommand>'. ---"
        )
        print("Usage examples:")
        print("  python main.py serve-api")
        print("  python main.py cocoindex setup")
        print("  python main.py cocoindex update")
        print("  python main.py cocoindex ls")
    logger.info(f"--- Script {__file__} (RAG API) __main__ block finished. ---")
