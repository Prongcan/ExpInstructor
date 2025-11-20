from sentence_transformers import SentenceTransformer
import numpy as np

# Global model cache
_model_cache = {}
_model_initialized = False

def _get_model(model_name: str = "BAAI/bge-m3"):
    """Get BGE-M3 model instance (with cache)"""
    if model_name not in _model_cache:
        try:
            import torch
            # Set environment variables to avoid meta tensor issues
            import os
            os.environ['TRANSFORMERS_OFFLINE'] = '0'
            
            # Force the use of GPU 2, as GPU 3 has insufficient memory
            if torch.cuda.is_available() and torch.cuda.device_count() > 2:
                device = 'cuda:2'  # Force the use of GPU 2
            elif torch.cuda.is_available():
                device = 'cuda'  # Fallback to the default GPU
            else:
                device = 'cpu'
            print(f"Using device: {device}")
            
            # Use trust_remote_code=True and low_cpu_mem_usage=True
            model = SentenceTransformer(
                model_name, 
                device=device,
                trust_remote_code=True,
                model_kwargs={'low_cpu_mem_usage': True}
            )
            
            # Ensure the model is on the correct device
            if hasattr(model, 'to'):
                model = model.to(device)
            
            _model_cache[model_name] = model
            print(f"BGE-M3 model {model_name} loaded successfully on device: {device}")
        except Exception as e:
            print(f"BGE-M3 model failed to load: {e}")
            # Fall back to CPU if GPU loading fails
            try:
                print("GPU loading failed, trying CPU...")
                model = SentenceTransformer(
                    model_name, 
                    device='cpu',
                    trust_remote_code=True,
                    model_kwargs={'low_cpu_mem_usage': True}
                )
                _model_cache[model_name] = model
                print(f"BGE-M3 model {model_name} loaded successfully (CPU fallback)")
            except Exception as e2:
                print(f"BGE-M3 model completely failed to load: {e2}")
                raise e2
    return _model_cache[model_name]

def embed_texts(texts: list[str], model_name: str = "BAAI/bge-m3") -> list[list[float]]:
    """Batch text embeddings. Returns an embedding list matching the input order."""
    model = _get_model(model_name)
    # Encode the list of texts with the BGE-M3 model
    embeddings = model.encode(texts)
    # Convert to list format and keep the same order as the inputs
    return [embedding.tolist() for embedding in embeddings]


def calculate_similarity(embeddings1: list[list[float]], embeddings2: list[list[float]] = None) -> np.ndarray:
    """Compute the similarity matrix between embeddings"""
    model = _get_model()
    embeddings1 = np.array(embeddings1)
    
    if embeddings2 is None:
        # Compute the self-similarity matrix
        similarities = model.similarity(embeddings1, embeddings1)
    else:
        # Compute the similarity between two embedding sets
        embeddings2 = np.array(embeddings2)
        similarities = model.similarity(embeddings1, embeddings2)
    
    return similarities

def preload_model(model_name: str = "BAAI/bge-m3"):
    """Preload the model to ensure it only loads once"""
    global _model_initialized
    if not _model_initialized:
        print(f"Preloading BGE-M3 model: {model_name}")
        _get_model(model_name)
        _model_initialized = True
        print("BGE-M3 model preloading finished")


if __name__ == "__main__":
    # Test code
    sentences = [
        "That is a happy person",
        "That is a happy dog", 
        "That is a very happy person",
        "Today is a sunny day"
    ]
    
    # Get embeddings
    embeddings = embed_texts(sentences)
    print(f"Embeddings shape: {len(embeddings)} x {len(embeddings[0])}")
    
    # Compute the similarity matrix
    similarities = calculate_similarity(embeddings)
    print(f"Similarities shape: {similarities.shape}")
    print(similarities)
