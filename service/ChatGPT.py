from openai import OpenAI
import os


def _get_client():
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_APIKEY")
    # Fall back to the original hard-coded value (not recommended, kept for compatibility)
    if not api_key:
        api_key = "YOUR_OPENAI_API_KEY"
    return OpenAI(api_key=api_key)


def chat(prompt, pdf_url, model: str = "gpt-4o"):
    client = _get_client()
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt,
                    },
                    {
                        "type": "input_file",
                        "file_url": pdf_url,
                    },
                ],
            }
        ],
    )
    return response.output_text


def chat_simple(prompt: str, model: str = "gpt-4o") -> str:
    client = _get_client()
    response = client.responses.create(
        model=model,
        temperature = 0.1,
        input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
    )
    return response.output_text


def embed_texts(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    """Batch text embeddings. Returns an embedding list matching the input order."""
    client = _get_client()
    # OpenAI embeddings API expects list of inputs
    resp = client.embeddings.create(model=model, input=texts)
    # resp.data[i].embedding -> list[float]
    return [item.embedding for item in resp.data]