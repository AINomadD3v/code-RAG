import asyncio
import os
import sys

import cocoindex
from dotenv import load_dotenv

# --- Global State ---
_IS_COCOINDEX_INITIALIZED = False
_code_query_handler = None  # Will hold the initialized QueryHandler

# Load environment variables from .env file
load_dotenv()
print(f"--- Running {__file__} (Module Level) ---")


# --- Explicit CocoIndex Initialization (CRUCIAL - Call this ONCE) ---
def initialize_cocoindex_globally():
    """
    Initializes CocoIndex settings. Should be called once.
    This is separate to allow controlled initialization when importing as a module.
    """
    global _IS_COCOINDEX_INITIALIZED
    if _IS_COCOINDEX_INITIALIZED:
        # print("CocoIndex already initialized.") # Optional: for debugging
        return

    print("--- Attempting explicit cocoindex.init() ---")
    db_url_init = os.getenv("COCOINDEX_DATABASE_URL")
    if not db_url_init:
        raise ValueError(
            "COCOINDEX_DATABASE_URL environment variable not set! Please set it in your .env file."
        )

    try:
        cocoindex.init(
            cocoindex.Settings(
                database=cocoindex.DatabaseConnectionSpec(url=db_url_init)
            )
        )
        _IS_COCOINDEX_INITIALIZED = True
        print("--- cocoindex.init() successful ---")
    except Exception as e:
        print(f"❌ ERROR: cocoindex.init() failed: {e}")
        _IS_COCOINDEX_INITIALIZED = (
            False  # Ensure it's marked as not initialized on failure
        )
        raise  # Re-raise the exception to halt if init fails


# Call initialization when the module is first loaded if it's going to be used by CLI or imported
# For CLI, @main_fn might also trigger init, but explicit is safer given past issues.
# For library use, the first call to a search function will ensure initialization.
initialize_cocoindex_globally()


# --- Custom CocoIndex Operations ---
@cocoindex.op.function()
def extract_extension(filename: str) -> str:
    """Extract the extension of a filename."""
    return os.path.splitext(filename)[1]


def code_to_embedding_logic(text: cocoindex.DataSlice) -> cocoindex.DataSlice:
    """Embeds text using SentenceTransformer model."""
    return text.transform(
        cocoindex.functions.SentenceTransformerEmbed(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
    )


# --- CocoIndex Flow Definition ---
# The decorator transforms 'code_embedding_flow_definition_func' into a Flow object.
# This Flow object is then registered with CocoIndex under the name "CodeEmbedding".
@cocoindex.flow_def(name="CodeEmbedding")
def code_embedding_flow_definition_func(  # The actual function cocoindex calls to build the flow
    flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope
):
    """Defines the data processing flow for code embeddings."""
    # This print confirms the framework is processing the flow definition
    print(f"--- Executing flow definition logic for 'CodeEmbedding' ---")

    data_scope["files"] = flow_builder.add_source(
        cocoindex.sources.LocalFile(
            path="uiautomator2",
            included_patterns=["*.py"],
            excluded_patterns=[".*", "**/__pycache__/**", "**/tests/**"],
        )
    )
    code_embeddings_collector = (
        data_scope.add_collector()
    )  # Changed variable name for clarity
    with data_scope["files"].row() as file:
        file["extension"] = file["filename"].transform(extract_extension)
        file["chunks"] = file["content"].transform(
            cocoindex.functions.SplitRecursively(),
            language=file[
                "extension"
            ],  # Consider mapping e.g. ".py" to "python" if needed by SplitRecursively
            chunk_size=300,
            chunk_overlap=50,
        )
        with file["chunks"].row() as chunk:
            chunk["embedding"] = chunk["text"].call(code_to_embedding_logic)
            code_embeddings_collector.collect(
                filename=file["filename"],
                location=chunk["location"],
                text=chunk["text"],
                embedding=chunk["embedding"],
            )
    code_embeddings_collector.export(
        "code_embeddings",  # This is the target_name for the QueryHandler
        cocoindex.storages.Postgres(),
        primary_key_fields=["filename", "location"],
        vector_indexes=[
            cocoindex.VectorIndexDef(
                field_name="embedding",
                metric=cocoindex.VectorSimilarityMetric.COSINE_SIMILARITY,
            )
        ],
    )
    print(f"--- Flow 'CodeEmbedding' definition processing complete ---")


# --- Query Handler Setup ---
def _get_or_create_query_handler():
    """Creates and returns the query handler, ensuring init and flow object retrieval."""
    global _code_query_handler
    if _code_query_handler:
        return _code_query_handler

    if not _IS_COCOINDEX_INITIALIZED:
        # This should ideally not be hit if initialize_cocoindex_globally() was called on import,
        # or if ensure_cocoindex_ready() is called before search.
        print(
            "WARNING: CocoIndex not initialized before query handler creation. Attempting init."
        )
        initialize_cocoindex_globally()

    # The flow object is created when @cocoindex.flow_def decorates code_embedding_flow_definition_func.
    # We can retrieve it by its registered name.
    try:
        flow_object = cocoindex.flow.get_flow_by_name("CodeEmbedding")
        if flow_object is None:
            raise ValueError(
                "Flow 'CodeEmbedding' not found in CocoIndex registry. Has 'setup' run after code changes?"
            )
    except Exception as e:
        print(f"Error getting flow 'CodeEmbedding' for query handler: {e}")
        print(
            "Please ensure 'python main.py cocoindex setup' has been run successfully."
        )
        raise  # Cannot proceed without the flow object

    def query_text_to_embedding(text: cocoindex.DataSlice) -> cocoindex.DataSlice:
        """Transforms a query string DataSlice into an embedding DataSlice."""
        return text.transform(
            cocoindex.functions.SentenceTransformerEmbed(
                model="sentence-transformers/all-MiniLM-L6-v2"  # Must match the indexing model
            )
        )

    _code_query_handler = cocoindex.query.SimpleSemanticsQueryHandler(
        name="CodeSearch",  # A name for this query handler configuration
        flow=flow_object,  # The flow object that created the index
        target_name="code_embeddings",  # The 'name' from the collector's export()
        query_transform_flow=query_text_to_embedding,
        default_similarity_metric=cocoindex.VectorSimilarityMetric.COSINE_SIMILARITY,
    )
    print("--- Query handler 'CodeSearch' initialized. ---")
    return _code_query_handler


# --- Public Search API for External Scripts ---
async def search_codebase_async(query_text: str, top_k: int = 5) -> list:
    """
    Asynchronously searches the indexed uiautomator2 codebase.
    This is the primary function to be called by your LLM assistant script.
    """
    # Ensure initialization and query handler setup
    # initialize_cocoindex_globally() # Called at module load
    query_handler = _get_or_create_query_handler()

    print(f"Asynchronously searching for: '{query_text}' (top_k={top_k})")
    results, _ = await query_handler.search_async(query_text, top_k=top_k)

    processed_results = []
    if results:
        for r in results:
            processed_results.append(
                {
                    "filename": r.data.get("filename"),
                    "location": str(r.data.get("location")),  # Convert range to string
                    "text": r.data.get("text"),
                    "score": r.score,
                }
            )
    return processed_results


def search_codebase(query_text: str, top_k: int = 5) -> list:
    """
    Synchronously searches the indexed uiautomator2 codebase.
    A wrapper around the async version for easier calling from sync code.
    """
    # Ensure an event loop exists if called from a sync context that might need one
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If we're already in a running loop (e.g., nested async call, Jupyter)
            # A more robust solution for nested asyncio might be needed if this is common.
            # For simplicity, this might still work or you might need to use ensure_future.
            # This specific scenario can be tricky.
            # For now, let's assume if a loop is running, it's okay to create a task.
            # However, asyncio.run() cannot be called when the event loop is already running.
            # So, this path needs careful consideration depending on the calling context.
            # A simpler approach for a sync wrapper when an outer loop might exist
            # is to use a thread or a more complex async bridge.
            # For now, let's assume typical script usage where a new loop for asyncio.run is fine.
            pass  # Let asyncio.run below handle it or error if nested incorrectly.
    except RuntimeError:  # No running event loop
        pass
    return asyncio.run(search_codebase_async(query_text, top_k))


# --- CocoIndex CLI Integration & Direct Run Behavior ---
@cocoindex.main_fn()
async def _cocoindex_main_entrypoint():  # Renamed for clarity
    """
    This function is targeted by @cocoindex.main_fn.
    Its body executes if `python main.py` is run without cocoindex subcommands.
    For cocoindex subcommands, the decorator handles CLI dispatch.
    """
    # initialize_cocoindex_globally() # Called at module load

    # This print helps understand when this function's body is actually run
    print(f"--- _cocoindex_main_entrypoint() executed. Args: {sys.argv} ---")

    is_cocoindex_command = len(sys.argv) > 1 and sys.argv[1] == "cocoindex"
    is_server_command = (
        is_cocoindex_command and len(sys.argv) > 2 and sys.argv[2] == "server"
    )

    if is_server_command:
        # The server command itself is handled by cocoindex framework.
        # We just need to keep the script alive.
        print("--- Server command detected. Keeping process alive for server... ---")
        await asyncio.Event().wait()
    elif not is_cocoindex_command:
        # Script was run directly, e.g., `python main.py`
        print("--- Script run directly. Interactive query test mode: ---")
        print("--- Make sure you have run 'python main.py cocoindex update' first! ---")
        query_handler = _get_or_create_query_handler()  # Ensure handler is ready
        while True:
            try:
                query_text = input(
                    "Enter search query for uiautomator2 code (or Enter to quit): "
                )
                if not query_text:
                    break
                results = await search_codebase_async(
                    query_text, top_k=3
                )  # Use the async version
                print("\nSearch results:")
                if results:
                    for i, result in enumerate(results):
                        print(f"\nResult {i+1} (Score: {result['score']:.4f}):")
                        print(f"  File: {result['filename']}")
                        print(f"  Location: {result['location']}")
                        print(
                            f"  Text:\n    {result['text'].strip().replace(chr(10), chr(10) + '    ')}"
                        )
                        print("---")
                else:
                    print("No results found.")
                print()
            except KeyboardInterrupt:
                print("\nExiting query mode.")
                break
            except Exception as e:
                print(f"Error during query: {e}")
    else:
        # For other cocoindex CLI commands (ls, setup, update), they do their work and exit.
        # This part of _cocoindex_main_entrypoint might be reached after they complete if they don't sys.exit().
        print(f"--- CocoIndex CLI command '{' '.join(sys.argv[1:])}' processed. ---")


# This standard block allows the script to be run for CLI commands or direct execution.
if __name__ == "__main__":
    # initialize_cocoindex_globally() # Called at module load
    # The @cocoindex.main_fn decorator on _cocoindex_main_entrypoint handles CLI parsing.
    # asyncio.run() executes the main async function.
    asyncio.run(_cocoindex_main_entrypoint())
    print(f"--- Script {__file__} __main__ block finished. ---")
