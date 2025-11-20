import sys
import os
import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import traceback
from sklearn.metrics.pairwise import cosine_similarity

# Add parent directory to path to import service
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from service import BGE_M3

def setup_logging(log_file: str = None) -> logging.Logger:
    """Configure logging for the retrieval system"""
    if log_file is None:
        log_file = f"retrieval_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

class EvidenceRetrievalSystem:
    """Evidence sentence retrieval system"""
    
    def __init__(self, embeddings_dir: str, logger: logging.Logger = None):
        self.embeddings_dir = embeddings_dir
        self.logger = logger or setup_logging()
        self.embeddings = {}
        self.embedding_index = {}
        self.model = None  # Cached model instance
        self.load_embeddings()
        self._load_model()  # Load the model during initialization
    
    def load_embeddings(self):
        """Load all embedding data"""
        try:
            # Load embedding index
            index_file = os.path.join(self.embeddings_dir, 'embedding_index.json')
            if os.path.exists(index_file):
                with open(index_file, 'r', encoding='utf-8') as f:
                    self.embedding_index = json.load(f)
                self.logger.info(f"Loaded embedding index with {len(self.embedding_index)} entries")
            
            # Load all embedding files
            embeddings_folder = os.path.join(self.embeddings_dir, 'embeddings')
            if os.path.exists(embeddings_folder):
                for filename in os.listdir(embeddings_folder):
                    if filename.endswith('_embeddings.json'):
                        file_path = os.path.join(embeddings_folder, filename)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            file_embeddings = json.load(f)
                        
                        for embedding_data in file_embeddings:
                            sentence_id = embedding_data['sentence_id']
                            self.embeddings[sentence_id] = {
                                'sentence_id': sentence_id,
                                'paper_id': embedding_data['paper_id'],
                                'review_id': embedding_data['review_id'],
                                'source_name': embedding_data['source_name'],
                                'target_name': embedding_data['target_name'],
                                'relationship': embedding_data['relationship'],
                                'evidence': embedding_data['evidence'],
                                'embedding': np.array(embedding_data['embedding']),
                                'embedding_dim': embedding_data['embedding_dim']
                            }
                
                self.logger.info(f"Loaded {len(self.embeddings)} embeddings")
            else:
                self.logger.warning(f"Embeddings folder not found: {embeddings_folder}")
                
        except Exception as e:
            self.logger.error(f"Error while loading embeddings: {str(e)}")
            self.logger.error(f"Details: {traceback.format_exc()}")
    
    def _load_model(self):
        """Load the BGE-M3 model once"""
        try:
            self.logger.info("Loading BGE-M3 model...")
            self.model = BGE_M3._get_model("BAAI/bge-m3")
            self.logger.info("BGE-M3 model loaded")
        except Exception as e:
            self.logger.error(f"Error loading BGE-M3 model: {str(e)}")
            self.model = None
    
    def get_query_embedding(self, query_text: str) -> Optional[np.ndarray]:
        """Encode the query text into an embedding"""
        try:
            if self.model is None:
                self.logger.error("BGE-M3 model not loaded")
                return None
            
            # Encode with the cached model
            embeddings = self.model.encode([query_text])
            if embeddings is not None and len(embeddings) > 0:
                return np.array(embeddings[0])
            return None
        except Exception as e:
            self.logger.error(f"Failed to create query embedding: {str(e)}")
            return None
    
    def cosine_similarity_search(self, query_text: str, top_k: int = 10) -> List[Dict]:
        """Retrieve sentences via cosine similarity"""
        try:
            # Build query embedding
            query_embedding = self.get_query_embedding(query_text)
            if query_embedding is None:
                self.logger.error("Unable to generate query embedding")
                return []
            
            # Compute similarities
            similarities = []
            for sentence_id, data in self.embeddings.items():
                embedding = data['embedding']
                similarity = cosine_similarity([query_embedding], [embedding])[0][0]
                similarities.append({
                    'sentence_id': sentence_id,
                    'paper_id': data['paper_id'],
                    'review_id': data['review_id'],
                    'source_name': data['source_name'],
                    'target_name': data['target_name'],
                    'relationship': data['relationship'],
                    'evidence': data['evidence'],
                    'similarity': float(similarity)
                })
            
            # Sort by similarity
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            return similarities[:top_k]
            
        except Exception as e:
            self.logger.error(f"Cosine similarity search failed: {str(e)}")
            self.logger.error(f"Details: {traceback.format_exc()}")
            return []
    
    def semantic_search(self, query_text: str, top_k: int = 10, 
                       similarity_threshold: float = 0.7) -> List[Dict]:
        """Semantic search with a similarity threshold"""
        try:
            results = self.cosine_similarity_search(query_text, top_k * 2)  # Fetch more for filtering
            
            # Filter low-similarity hits
            filtered_results = [r for r in results if r['similarity'] >= similarity_threshold]
            
            return filtered_results[:top_k]
            
        except Exception as e:
            self.logger.error(f"Semantic search failed: {str(e)}")
            self.logger.error(f"Details: {traceback.format_exc()}")
            return []
    
    def search_by_paper_id(self, paper_id: str) -> List[Dict]:
        """Retrieve evidence by paper ID"""
        try:
            results = []
            for sentence_id, data in self.embeddings.items():
                if data['paper_id'] == paper_id:
                    results.append({
                        'sentence_id': sentence_id,
                        'paper_id': data['paper_id'],
                        'review_id': data['review_id'],
                        'source_name': data['source_name'],
                        'target_name': data['target_name'],
                        'relationship': data['relationship'],
                        'evidence': data['evidence'],
                        'similarity': 1.0  # Exact match
                    })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Paper ID search failed: {str(e)}")
            return []
    
    def search_by_relationship(self, relationship: str, top_k: int = 10) -> List[Dict]:
        """Retrieve evidence by relationship type"""
        try:
            results = []
            for sentence_id, data in self.embeddings.items():
                if relationship.lower() in data['relationship'].lower():
                    results.append({
                        'sentence_id': sentence_id,
                        'paper_id': data['paper_id'],
                        'review_id': data['review_id'],
                        'source_name': data['source_name'],
                        'target_name': data['target_name'],
                        'relationship': data['relationship'],
                        'evidence': data['evidence'],
                        'similarity': 1.0  # Exact match
                    })
            
            # Sort by similarity (all 1.0, effectively alphabetical)
            results.sort(key=lambda x: x['sentence_id'])
            
            return results[:top_k]
            
        except Exception as e:
            self.logger.error(f"Relationship search failed: {str(e)}")
            return []
    
    def search_by_keywords(self, keywords: List[str], top_k: int = 10) -> List[Dict]:
        """Retrieve evidence using a keyword bag query"""
        try:
            # Merge keywords into a single query
            query_text = " ".join(keywords)
            return self.cosine_similarity_search(query_text, top_k)
            
        except Exception as e:
            self.logger.error(f"Keyword search failed: {str(e)}")
            return []
    
    def get_evidence_by_sentence_id(self, sentence_id: str) -> Optional[Dict]:
        """Fetch evidence metadata by sentence_id"""
        return self.embeddings.get(sentence_id)
    
    def get_statistics(self) -> Dict:
        """Return system statistics"""
        if not self.embeddings:
            return {
                'total_embeddings': 0,
                'total_papers': 0,
                'total_reviews': 0,
                'embedding_dim': 0
            }
        
        papers = set(data['paper_id'] for data in self.embeddings.values())
        reviews = set(f"{data['paper_id']}_{data['review_id']}" for data in self.embeddings.values())
        
        return {
            'total_embeddings': len(self.embeddings),
            'total_papers': len(papers),
            'total_reviews': len(reviews),
            'embedding_dim': list(self.embeddings.values())[0]['embedding_dim']
        }

def demo_retrieval_system():
    """Demonstrate retrieval system capabilities"""
    logger = setup_logging()
    
    # Initialize retrieval system
    embeddings_dir = 'RAG_baseline_review_sentence'
    retrieval_system = EvidenceRetrievalSystem(embeddings_dir, logger)
    
    # Display stats
    stats = retrieval_system.get_statistics()
    logger.info(f"Retrieval system statistics: {stats}")
    
    # Example queries
    test_queries = [
        "machine learning performance evaluation",
        "neural network architecture",
        "deep learning optimization",
        "reinforcement learning algorithms",
        "experimental validation",
        "model training efficiency"
    ]
    
    for query in test_queries:
        logger.info(f"\nQuery: {query}")
        results = retrieval_system.cosine_similarity_search(query, top_k=5)
        
        for i, result in enumerate(results, 1):
            logger.info(f"{i}. Paper: {result['paper_id']}, Review: {result['review_id']}")
            logger.info(f"   Relationship: {result['source_name']} --[{result['relationship']}]--> {result['target_name']}")
            logger.info(f"   Similarity: {result['similarity']:.4f}")
            logger.info(f"   Evidence: {result['evidence'][:200]}...")
            logger.info("")

def main():
    demo_retrieval_system()

if __name__ == "__main__":
    main()
