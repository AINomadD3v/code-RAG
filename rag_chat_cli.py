# rag_chat_cli.py
import asyncio
import json
import logging
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel  # Using Pydantic for message structures

# --- Configuration ---
load_dotenv()  # Load .env file for API keys

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

COCOINDEX_SEARCH_API_URL = os.getenv(
    "COCOINDEX_SEARCH_API_URL", "http://localhost:8000/search"
)
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_DEFAULT_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

if not DEEPSEEK_API_KEY:
    logger.warning(
        "DEEPSEEK_API_KEY not found in environment variables. "
        "LLM service will not function correctly."
    )


# --- Pydantic Models for LLM Interaction (adapted from your llm_service.py) ---
class ToolCallFunction(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


class ToolCall(BaseModel):
    id: Optional[str] = None
    type: str = "function"
    function: ToolCallFunction


class ChatMessageContent(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


# --- RAG Context Retrieval ---
async def fetch_rag_context(query: str, top_k: int = 3) -> str:
    """
    Fetches relevant code snippets from the CocoIndex RAG API.
    """
    if not COCOINDEX_SEARCH_API_URL:
        logger.error("COCOINDEX_SEARCH_API_URL is not configured.")
        return "Error: RAG service URL not configured."

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            logger.info(f"Querying RAG API: {COCOINDEX_SEARCH_API_URL} for '{query}'")
            response = await client.get(
                COCOINDEX_SEARCH_API_URL, params={"query": query, "top_k": top_k}
            )
            response.raise_for_status()
            search_data = response.json()

            results = search_data.get("results", [])
            if not results:
                return "No specific code snippets found in the uiautomator2 codebase for this query."

            context_str = "Relevant uiautomator2 code snippets:\n\n"
            for i, snippet in enumerate(results):
                context_str += f"Snippet {i+1} (from {snippet.get('filename', 'N/A')}, score: {snippet.get('score', 0.0):.2f}):\n"
                context_str += "```python\n"
                context_str += f"{snippet.get('text', '')}\n"
                context_str += "```\n\n"
            return context_str.strip()
    except httpx.RequestError as e:
        logger.error(f"RAG API RequestError: {e}")
        return f"Error: Could not connect to the code search service: {e}"
    except httpx.HTTPStatusError as e:
        logger.error(
            f"RAG API HTTPStatusError: {e.response.status_code} - {e.response.text}"
        )
        return f"Error: Code search service returned an error: {e.response.status_code}"
    except Exception as e:
        logger.error(f"Unexpected error fetching RAG context: {e}", exc_info=True)
        return "Error: An unexpected error occurred while retrieving code context."


# --- DeepSeek LLM Interaction ---
def _build_deepseek_payload_messages(
    user_prompt: str,
    rag_context: Optional[str],  # RAG context is now optional
    history: List[ChatMessageContent],
    system_prompt_override: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Constructs the 'messages' list for the DeepSeek API payload,
    incorporating RAG context if available.
    """
    system_prompt_content = system_prompt_override or (
        "You are an expert Python automation assistant specializing in the uiautomator2 library for Android testing. "
        "You interact with an Android device object typically named `d`. "
        "Your primary goal is to provide accurate uiautomator2 Python code snippets and concise explanations.\n\n"
        "RULES:\n"
        "1. ALWAYS wrap Python code in triple backticks (```python ... ```).\n"
        "2. Prioritize using information from the 'Relevant uiautomator2 code snippets' context if provided. This context is retrieved from the actual uiautomator2 codebase.\n"
        "3. If the provided context helps answer the question, explicitly state that you are using it.\n"
        "4. If the context is not relevant or insufficient, rely on your general knowledge of uiautomator2.\n"
        "5. If you are unsure or the request is ambiguous, ask for clarification rather than hallucinating.\n"
        "6. Keep explanations brief (1-2 sentences) and provide them *after* the code block.\n"
        "7. If generating a function, include a simple example call to it (e.g., `my_function(d)`) unless asked not to.\n"
        "8. Focus on uiautomator2 syntax. Be precise and tactical."
    )

    messages_for_api: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt_content}
    ]

    for msg_content_model in history:
        messages_for_api.append(
            msg_content_model.model_dump(exclude_none=True, by_alias=True)
        )

    # Construct the current user message
    full_user_content = ""
    if (
        rag_context
        and "Error:" not in rag_context
        and "No specific code snippets found" not in rag_context
    ):
        full_user_content += (
            f"## Context from uiautomator2 Codebase:\n{rag_context}\n\n"
        )

    full_user_content += f"## User Question:\n{user_prompt}"

    messages_for_api.append({"role": "user", "content": full_user_content})
    # logger.debug(f"Messages for LLM API: {json.dumps(messages_for_api, indent=2)}")
    return messages_for_api


async def stream_deepseek_completion(
    messages: List[Dict[str, Any]],
    model: Optional[str] = None,
    temperature: Optional[float] = 0.7,
    max_tokens: Optional[int] = 2048,
) -> AsyncGenerator[str, None]:
    """
    Streams chat completions from the DeepSeek API.
    Yields content chunks (text or tool call information).
    """
    if not DEEPSEEK_API_KEY:
        yield "Error: DeepSeek API key is not configured.\n"
        return

    payload = {
        "model": model or DEEPSEEK_DEFAULT_MODEL,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    payload = {k: v for k, v in payload.items() if v is not None}  # Remove None values

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            logger.info(
                f"Streaming request to DeepSeek API. Model: {payload.get('model')}"
            )
            async with client.stream(
                "POST", DEEPSEEK_API_URL, json=payload, headers=headers
            ) as response:
                if response.status_code != 200:
                    error_content_bytes = await response.aread()
                    error_content_str = error_content_bytes.decode(errors="replace")
                    logger.error(
                        f"DeepSeek API Error: {response.status_code} - {error_content_str}"
                    )
                    yield f"Error from LLM API ({response.status_code}): {error_content_str}\n"
                    return

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    if line == "data: [DONE]":
                        logger.info("DeepSeek stream finished with [DONE].")
                        break
                    if line.startswith("data: "):
                        json_data_str = line[len("data: ") :]
                        try:
                            chunk_data = json.loads(json_data_str)
                            choice = chunk_data.get("choices", [{}])[0]
                            delta = choice.get("delta", {})
                            delta_content_text = delta.get("content")
                            if delta_content_text:
                                yield delta_content_text
                            # Basic tool call detection (can be expanded)
                            if choice.get(
                                "finish_reason"
                            ) == "tool_calls" and delta.get("tool_calls"):
                                yield f"\n[Tool Call Requested by LLM: {json.dumps(delta.get('tool_calls'))}]\n"
                        except json.JSONDecodeError:
                            logger.error(
                                f"Error parsing LLM stream chunk: '{json_data_str}'"
                            )
        except httpx.RequestError as e:
            logger.error(f"Network issue contacting DeepSeek: {e}", exc_info=True)
            yield f"Error: Network issue with LLM provider: {e}\n"
        except Exception as e:
            logger.error(f"Unexpected error during DeepSeek stream: {e}", exc_info=True)
            yield f"Error: Unexpected error with LLM service: {e}\n"


# --- Main CLI Chat Loop ---
async def chat_loop():
    """Main interactive chat loop for the RAG CLI assistant."""
    print("RAG CLI Assistant for uiautomator2 (powered by CocoIndex & DeepSeek)")
    print(f"Using CocoIndex RAG API: {COCOINDEX_SEARCH_API_URL}")
    print(f"Using DeepSeek Model: {DEEPSEEK_DEFAULT_MODEL}")
    print("Type 'exit' or 'quit' to end.")
    print("-" * 30)

    conversation_history: List[ChatMessageContent] = []

    while True:
        try:
            user_input = await asyncio.to_thread(input, "You: ")  # Async input
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting chat.")
                break
            if not user_input.strip():
                continue

            # 1. Get RAG context from CocoIndex API
            print("Fetching context from uiautomator2 codebase...")
            rag_context = await fetch_rag_context(user_input, top_k=3)
            if (
                "Error:" not in rag_context
                and "No specific code snippets found" not in rag_context
            ):
                print(
                    f"\n--- Context Retrieved ---\n{rag_context}\n-------------------------\n"
                )
            else:
                print(f"\n--- RAG Info: {rag_context} ---\n")

            # 2. Prepare messages for DeepSeek
            # Add user's current message to history for the call, but don't permanently store RAG context in history
            current_turn_user_message = ChatMessageContent(
                role="user", content=user_input
            )

            # Build messages for this specific turn, including RAG context
            # The RAG context will be prepended to the user's *current* prompt within _build_deepseek_payload_messages
            messages_for_llm = _build_deepseek_payload_messages(
                user_prompt=user_input,
                rag_context=rag_context,  # Pass the fetched RAG context
                history=conversation_history,  # Pass the ongoing history
            )

            # 3. Stream response from DeepSeek
            print("Assistant: ", end="", flush=True)
            assistant_response_buffer = ""
            async for chunk in stream_deepseek_completion(messages_for_llm):
                print(chunk, end="", flush=True)
                assistant_response_buffer += chunk
            print()  # Newline after assistant's full response

            # 4. Update history
            conversation_history.append(
                current_turn_user_message
            )  # Add user's actual message
            if (
                assistant_response_buffer.strip()
            ):  # Add assistant's response if not empty
                conversation_history.append(
                    ChatMessageContent(
                        role="assistant", content=assistant_response_buffer.strip()
                    )
                )

            # Limit history size to avoid overly long contexts for the LLM
            if len(conversation_history) > 10:  # Keep last 5 pairs (10 messages)
                conversation_history = conversation_history[-10:]

        except KeyboardInterrupt:
            print("\nExiting chat.")
            break
        except Exception as e:
            logger.error(f"Error in chat loop: {e}", exc_info=True)
            print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    if not DEEPSEEK_API_KEY:
        print("CRITICAL: DEEPSEEK_API_KEY is not set. The application cannot run.")
        print("Please set it in your .env file or as an environment variable.")
    else:
        try:
            asyncio.run(chat_loop())
        except KeyboardInterrupt:
            print("\nApplication terminated by user.")
