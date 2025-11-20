# Retrive_Generate/graph_retrieval_system.py

import json
import networkx as nx
import re
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import math
import os
import hashlib
from pathlib import Path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from service import ChatGPT
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from tqdm import tqdm
except Exception:
    SentenceTransformer = None
    np = None
    tqdm = None

# Import embedding services
import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Prefer BGE-M3 and fall back to ChatGPT
try:
    from service.BGE_M3 import embed_texts as embed_texts_bge_m3
    BGE_M3_AVAILABLE = True
except ImportError:
    BGE_M3_AVAILABLE = False
    print("BGE-M3 unavailable, falling back to ChatGPT embeddings")

try:
    from service.ChatGPT import embed_texts as embed_texts_chatgpt
    CHATGPT_AVAILABLE = True
except ImportError:
    CHATGPT_AVAILABLE = False
    print("ChatGPT embeddings unavailable")

def embed_texts(texts: list[str], model: str = "BAAI/bge-m3") -> list[list[float]]:
    """
    Unified embedding interface that prefers BGE-M3 and falls back to ChatGPT
    
    Args:
        texts: List of texts
        model: Model name
        
    Returns:
        List of vectors
    """
    # Prefer BGE-M3
    if BGE_M3_AVAILABLE:
        try:
            return embed_texts_bge_m3(texts, model_name=model)
        except Exception as e:
            print(f"BGE-M3 embedding failed: {e}")
            if CHATGPT_AVAILABLE:
                print("Falling back to ChatGPT embeddings")
                return embed_texts_chatgpt(texts, model="text-embedding-3-small")
            else:
                raise RuntimeError("All embedding services are unavailable")
    
    # Use ChatGPT
    elif CHATGPT_AVAILABLE:
        return embed_texts_chatgpt(texts, model=model)
    
    else:
        raise RuntimeError("No embedding services available")

class GraphRetrievalSystem:
    def __init__(self, graph_file, gpu_device="auto", batch_size=256, build_index_immediately=True):
        """
        Initialize the graph retrieval system
        
        Args:
            graph_file: Path to the graph file
            gpu_device: GPU device such as "auto", "cuda:0", "cuda:1"
            batch_size: Batch size; larger batches can improve GPU utilization
            build_index_immediately: Whether to build the index right away, default True
        """
        self.G = nx.DiGraph()
        self.gpu_device = gpu_device
        # Force GPU 2 because GPU 3 has insufficient memory
        if gpu_device == "auto":
            import torch
            if torch.cuda.is_available() and torch.cuda.device_count() > 2:
                self.gpu_device = "cuda:2"
            else:
                self.gpu_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.gpu_device = gpu_device
        self.batch_size = batch_size
        self.load_graph(graph_file)
        if build_index_immediately:
            self.build_index()
    
    def load_graph(self, graph_file):
        """
        Load graph data from JSON. Supports two formats:
        1) { "nodes": [...], "edges": [...] }
        2) [ { "paper_id": "...", "review_id": "...", "edges": [ {...}, ... ] }, ... ]
        """
        with open(graph_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.G.clear()

        def add_edge_record(src_name, tgt_name, rel, evd, edge_meta=None):
            # Ensure nodes exist
            if not self.G.has_node(src_name):
                self.G.add_node(src_name, type='entity')
            if not self.G.has_node(tgt_name):
                self.G.add_node(tgt_name, type='entity')
            # Add the edge
            attrs = {
                "relationship": rel or "",
                "evidence": evd or "",
            }
            if edge_meta:
                attrs.update(edge_meta)
            self.G.add_edge(src_name, tgt_name, **attrs)

        # Case 1: legacy format
        if isinstance(data, dict) and "nodes" in data and "edges" in data:
            for node in data["nodes"]:
                self.G.add_node(node.get("name", node.get("id", "")),
                                id=node.get("id", ""),
                                type=node.get("type", "entity"))
            for edge in data["edges"]:
                add_edge_record(
                    edge.get("source"), edge.get("target"),
                    edge.get("relationship"), edge.get("evidence"),
                    {"edge_id": edge.get("id", "")}
                )

        # Case 2: all_graphs.json format (list where each entry has edges)
        elif isinstance(data, list):
            for item_idx, item in enumerate(data):
                paper_id = item.get("paper_id", "")
                review_id = item.get("review_id", "")
                edges_raw = item.get("edges", [])
                # Support edges defined as a list or a single dict
                if isinstance(edges_raw, dict):
                    edges_iter = [edges_raw]
                elif isinstance(edges_raw, list):
                    edges_iter = edges_raw
                else:
                    edges_iter = []
                for edge_idx, e in enumerate(edges_iter):
                    add_edge_record(
                        e.get("source_name"), e.get("target_name"),
                        e.get("relationship"), e.get("evidence"),
                        {
                            "edge_id": f"{paper_id}:{review_id}:{edge_idx}",
                            "paper_id": paper_id,
                            "review_id": review_id
                        }
                    )
        else:
            raise ValueError("Unsupported graph data format")

        print(f"Graph loaded: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")

    def build_index(self):
        """
        Build the search index
        """
        self.entity_index = {}
        self.relationship_index = defaultdict(list)
        self.evidence_index = defaultdict(list)
        self._entity_texts = []
        self._edge_texts = []
        self._entity_names = []
        self._edge_triplets = []
        self._entity_embeddings = None
        self._edge_embeddings = None
        self._embed_model = None
        
        # Build entity index
        for node in self.G.nodes():
            self.entity_index[node.lower()] = node
            self._entity_names.append(node)
            # Entity description text
            self._entity_texts.append(f"ENTITY: {node}")
        
        # Build relationship index
        for source, target, attrs in self.G.edges(data=True):
            rel = attrs.get('relationship', '')
            evidence = attrs.get('evidence', '')
            
            self.relationship_index[rel.lower()].append((source, target, attrs))
            self.evidence_index[evidence.lower()].append((source, target, attrs))
            # Edge description text (triplet plus evidence snippet)
            triplet_text = f"EDGE: {source} --[{rel}]--> {target}. EVIDENCE: {evidence}"
            self._edge_texts.append(triplet_text)
            self._edge_triplets.append((source, target, attrs))

        # Build vector indexes (if sentence-transformers is available)
        self._maybe_build_embeddings()

    def _maybe_build_embeddings(self, model_name: str = 'BAAI/bge-m3'):
        """
        Build semantic embeddings for entities and edges using the BGE-M3 or ChatGPT embedding API.
        """
        if not self._entity_texts and not self._edge_texts:
            return
        
        # Try loading from cache
        cache_dir, ent_path, edge_path, meta_path = self._cache_paths(model_name)
        signature = self._compute_signature()
        try:
            if ent_path.exists() and edge_path.exists() and meta_path.exists():
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                if meta.get('model_name') == model_name and meta.get('signature') == signature:
                    # Load cached embeddings
                    with open(ent_path, 'r', encoding='utf-8') as f:
                        self._entity_embeddings = json.load(f) if self._entity_texts else None
                    with open(edge_path, 'r', encoding='utf-8') as f:
                        self._edge_embeddings = json.load(f) if self._edge_texts else None
                    print(f"Loaded embedding cache: {cache_dir}")
                    return
        except Exception:
            # Ignore corrupted cache and rebuild
            pass

        # Recompute and save the cache
        try:
            if self._entity_texts:
                print(f"Generating embeddings for {len(self._entity_texts)} entities...")
                # Use a larger batch size for BGE-M3 to maximize GPU usage
                batch_size = self.batch_size if BGE_M3_AVAILABLE else 100
                self._entity_embeddings = self._batch_embed_texts(self._entity_texts, model_name, batch_size=batch_size)
            if self._edge_texts:
                print(f"Generating embeddings for {len(self._edge_texts)} edges...")
                batch_size = self.batch_size if BGE_M3_AVAILABLE else 100
                self._edge_embeddings = self._batch_embed_texts(self._edge_texts, model_name, batch_size=batch_size)

            cache_dir.mkdir(parents=True, exist_ok=True)
            if self._entity_embeddings is not None:
                with open(ent_path, 'w', encoding='utf-8') as f:
                    json.dump(self._entity_embeddings, f, ensure_ascii=False)
            if self._edge_embeddings is not None:
                with open(edge_path, 'w', encoding='utf-8') as f:
                    json.dump(self._edge_embeddings, f, ensure_ascii=False)
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'model_name': model_name,
                    'signature': signature,
                    'entity_count': len(self._entity_texts),
                    'edge_count': len(self._edge_texts)
                }, f, ensure_ascii=False, indent=2)
            print(f"Generated and cached embeddings: {cache_dir}")
        except Exception as e:
            print(f"Failed to generate/save embeddings: {e}")
            self._entity_embeddings = None
            self._edge_embeddings = None

    def _compute_signature(self) -> str:
        """
        Compute a signature of the current graph content based on node and edge text hashes; any change invalidates the cache.
        """
        h = hashlib.sha1()
        h.update(f"N:{len(self._entity_texts)} E:{len(self._edge_texts)}".encode('utf-8'))
        # Sample to avoid huge memory usage; can hash everything if needed
        for txt in self._entity_texts[:5000]:
            h.update(txt.encode('utf-8', errors='ignore'))
        for txt in self._edge_texts[:20000]:
            h.update(txt.encode('utf-8', errors='ignore'))
        return h.hexdigest()

    def _cache_paths(self, model_name: str):
        """
        Return the cache directory and file paths.
        Path: <repo>/Retrive_Generate/.embeddings/<model_hash>/{entities.json,edges.json,meta.json}
        """
        model_hash = hashlib.sha1(model_name.encode('utf-8')).hexdigest()[:12]
        base_dir = Path(__file__).resolve().parent / '.embeddings' / model_hash
        ent_path = base_dir / 'entities.json'
        edge_path = base_dir / 'edges.json'
        meta_path = base_dir / 'meta.json'
        return base_dir, ent_path, edge_path, meta_path

    def _batch_embed_texts(self, texts: List[str], model: str, batch_size: int = 100) -> List[List[float]]:
        """
        Batch embeddings to avoid API limits
        """
        if not texts:
            return []
        
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        # Create a progress bar
        if tqdm:
            pbar = tqdm(total=len(texts), desc="Generating embeddings", unit="text", 
                       bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
        else:
            pbar = None
        
        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                try:
                    # Filter empty or overly long texts
                    filtered_batch = []
                    for text in batch:
                        if text and len(text.strip()) > 0 and len(text) < 8000:  # OpenAI limit
                            filtered_batch.append(text.strip())
                        else:
                            # Use placeholders for empty or overly long texts
                            filtered_batch.append("empty")
                    
                    if filtered_batch:
                        batch_embeddings = embed_texts(filtered_batch, model=model)
                        all_embeddings.extend(batch_embeddings)
                    else:
                        # If the entire batch is filtered, add zero vectors (BGE-M3 dimension is 1024)
                        embedding_dim = 1024 if BGE_M3_AVAILABLE else 1536
                        all_embeddings.extend([[0.0] * embedding_dim] * len(batch))
                    
                    # Update progress bar
                    if pbar:
                        pbar.update(len(batch))
                        pbar.set_postfix({'batch': f"{batch_num}/{total_batches}"})
                    else:
                        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} texts)...")
                    
                    # BGE-M3 does not require delay; ChatGPT needs pauses to avoid rate limits
                    if not BGE_M3_AVAILABLE or not model.startswith("BAAI/bge-m3"):
                        import time
                        time.sleep(0.1)
                    
                except Exception as e:
                    print(f"Batch {batch_num} failed: {e}")
                    # Add zero vectors as placeholders (BGE-M3 dimension is 1024)
                    embedding_dim = 1024 if BGE_M3_AVAILABLE else 1536
                    all_embeddings.extend([[0.0] * embedding_dim] * len(batch))
                    if pbar:
                        pbar.update(len(batch))
        
        finally:
            if pbar:
                pbar.close()
        
        return all_embeddings

    def _cosine_topk(self, query_vec: List[float], matrix: List[List[float]], k: int = 10) -> List[int]:
        if query_vec is None or matrix is None or len(matrix) == 0:
            return []
        
        # Compute cosine similarity
        similarities = []
        for vec in matrix:
            # Dot product
            dot_product = sum(a * b for a, b in zip(query_vec, vec))
            # Vector magnitude
            query_norm = sum(a * a for a in query_vec) ** 0.5
            vec_norm = sum(a * a for a in vec) ** 0.5
            
            if query_norm == 0 or vec_norm == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (query_norm * vec_norm)
            similarities.append(similarity)
        
        # Get top-k indices
        indexed_sims = [(i, sim) for i, sim in enumerate(similarities)]
        indexed_sims.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in indexed_sims[:k]]

    def semantic_vector_search(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Perform semantic vector retrieval against both entity texts and edge texts.
        Returns the top-k candidates for entities and edges.
        """
        if self._entity_embeddings is None and self._edge_embeddings is None:
            self._maybe_build_embeddings()
        if self._entity_embeddings is None and self._edge_embeddings is None:
            return {"error": "Embedding model not ready"}

        # Build the query vector
        try:
            q_vec = embed_texts([query], model='BAAI/bge-m3')[0]
        except Exception as e:
            return {"error": f"Failed to generate query vector: {e}"}

        # Entity top-k
        entity_indices = self._cosine_topk(q_vec, self._entity_embeddings, k=top_k) if self._entity_embeddings is not None else []
        entity_hits = [self._entity_names[i] for i in entity_indices]

        # Edge top-k
        edge_indices = self._cosine_topk(q_vec, self._edge_embeddings, k=top_k) if self._edge_embeddings is not None else []
        edge_hits = [self._edge_triplets[i] for i in edge_indices]

        return {
            "query": query,
            "top_entities": entity_hits,
            "top_edges": edge_hits
        }
    
    def search_entities(self, query: str) -> List[str]:
        """
        Search entities
        """
        query = query.lower()
        results = []
        
        for entity in self.entity_index:
            if query in entity:
                results.append(self.entity_index[entity])
        
        return results
    
    def search_relationships(self, query: str) -> List[Tuple]:
        """
        Search relationships
        """
        query = query.lower()
        results = []
        
        for rel_type, edges in self.relationship_index.items():
            if query in rel_type:
                results.extend(edges)
        
        return results
    
    def search_evidence(self, query: str) -> List[Tuple]:
        """
        Search evidence
        """
        query = query.lower()
        results = []
        
        for evidence, edges in self.evidence_index.items():
            if query in evidence:
                results.extend(edges)
        
        return results
    
    def get_entity_relationships(self, entity: str) -> Dict[str, List]:
        """
        Get all relationships for a specific entity
        """
        if entity not in self.G:
            return {"error": f"Entity '{entity}' does not exist"}
        
        incoming = []
        outgoing = []
        
        for source, target, attrs in self.G.edges(data=True):
            if source == entity:
                outgoing.append({
                    "target": target,
                    "relationship": attrs.get('relationship', ''),
                    "evidence": attrs.get('evidence', '')
                })
            elif target == entity:
                incoming.append({
                    "source": source,
                    "relationship": attrs.get('relationship', ''),
                    "evidence": attrs.get('evidence', '')
                })
        
        return {
            "entity": entity,
            "outgoing_relationships": outgoing,
            "incoming_relationships": incoming,
            "total_connections": len(incoming) + len(outgoing)
        }
    
    def find_paths(self, source: str, target: str, max_length: int = 3) -> List[List[str]]:
        """
        Find paths between two entities
        """
        if source not in self.G or target not in self.G:
            return []
        
        try:
            paths = list(nx.all_simple_paths(self.G, source, target, cutoff=max_length))
            return paths
        except nx.NetworkXNoPath:
            return []
    
    def get_related_entities(self, entity: str, depth: int = 1) -> List[str]:
        """
        Get related entities
        """
        if entity not in self.G:
            return []
        
        related = set()
        current_level = {entity}
        
        for _ in range(depth):
            next_level = set()
            for node in current_level:
                # Add neighbor nodes
                for neighbor in self.G.neighbors(node):
                    if neighbor not in related:
                        related.add(neighbor)
                        next_level.add(neighbor)
                # Add predecessor nodes
                for predecessor in self.G.predecessors(node):
                    if predecessor not in related:
                        related.add(predecessor)
                        next_level.add(predecessor)
            current_level = next_level
        
        return list(related)
    
    def semantic_search(self, query: str) -> Dict[str, Any]:
        """
        Semantic search across entities, relationships, and evidence
        """
        query = query.lower()
        results = {
            "query": query,
            "entities": [],
            "relationships": [],
            "evidence": [],
            "suggestions": []
        }
        
        # Search entities
        entity_matches = self.search_entities(query)
        results["entities"] = entity_matches
        
        # Search relationships
        rel_matches = self.search_relationships(query)
        results["relationships"] = rel_matches
        
        # Search evidence
        evidence_matches = self.search_evidence(query)
        results["evidence"] = evidence_matches
        
        # Generate suggestions
        if entity_matches:
            for entity in entity_matches[:3]:  # Only take the first three entities
                rel_info = self.get_entity_relationships(entity)
                results["suggestions"].append({
                    "entity": entity,
                    "connections": rel_info["total_connections"]
                })
        
        return results

    def smart_search(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Prefer vector-based semantic retrieval; fall back to keyword search when unavailable.
        Returns the same structure as semantic_search for display reuse.
        """
        # Vector retrieval path
        if self._entity_embeddings is not None or self._edge_embeddings is not None:
            vec = self.semantic_vector_search(query, top_k=top_k)
            if 'error' not in vec:
                entities = vec.get('top_entities', [])
                edges = vec.get('top_edges', [])
                # Normalize into a compatible structure
                results = {
                    'query': query,
                    'entities': entities,
                    'relationships': edges,  # [(source, target, attrs), ...]
                    'evidence': edges,       # Reuse; display helpers pull only evidence fragments
                    'suggestions': []
                }
                # Suggestions based on entity connection counts
                for entity in entities[:3]:
                    info = self.get_entity_relationships(entity)
                    if 'error' not in info:
                        results['suggestions'].append({
                            'entity': entity,
                            'connections': info['total_connections']
                        })
                return results
        # Fallback to keyword search
        return self.semantic_search(query)
    
    def interactive_search(self):
        """
        Interactive search CLI
        """
        print("=== Graph Retrieval System ===")
        print("Available commands:")
        print("  search <keyword> - Semantic search (prefers vectors, auto fallback)")
        print("  ssearch <keyword> - Vector semantic search (uses BGE-M3)")
        print("  search_node_and_edge <keyword> - Combined node-edge-node search")
        print("  entity <name> - Show entity relationships")
        print("  path <source> <target> - Find a path")
        print("  related <name> - Find related entities")
        print("  stats - Display graph statistics")
        print("  quit - Exit")
        print()
        
        while True:
            try:
                command = input("Enter command: ").strip()
                
                if command.lower() == 'quit':
                    break
                elif command.lower() == 'stats':
                    self.show_stats()
                elif command.startswith('search '):
                    query = command[7:].strip()
                    self.display_search_results(self.smart_search(query))
                elif command.startswith('ssearch '):
                    query = command[8:].strip()
                    res = self.semantic_vector_search(query)
                    if 'error' in res:
                        print(f"Error: {res['error']}")
                    else:
                        print(f"\n=== Vector semantic search: '{res['query']}' ===")
                        if res['top_entities']:
                            print("\nTop entities:")
                            for i, ent in enumerate(res['top_entities'][:10]):
                                print(f"  {i+1}. {ent}")
                        if res['top_edges']:
                            print("\nTop relationships:")
                            for i, (s, t, attrs) in enumerate(res['top_edges'][:10]):
                                print(f"  {i+1}. {s} --[{attrs.get('relationship','')}]--> {t}")
                elif command.startswith('entity '):
                    entity = command[7:].strip()
                    self.display_entity_info(self.get_entity_relationships(entity))
                elif command.startswith('path '):
                    parts = command[5:].strip().split()
                    if len(parts) >= 2:
                        source, target = parts[0], parts[1]
                        self.display_paths(self.find_paths(source, target))
                    else:
                        print("Usage: path <source> <target>")
                elif command.startswith('related '):
                    entity = command[8:].strip()
                    related = self.get_related_entities(entity)
                    self.display_related_entities(entity, related)
                elif command.startswith('search_node_and_edge '):
                    query = command[18:].strip()
                    self.display_search_node_and_edge(self.search_similar_node_and_edge(query))
                else:
                    print("Unknown command, please try again")
                
                print()
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def parse_llm_json_output(self, chat_simple: str):
        """
        Parse a JSON object from LLM output.
        Supports:
        - Pure JSON text
        - JSON wrapped in ```json ... ``` or ``` ... ```
        - Extra descriptive text surrounding the JSON

        Args:
            chat_simple (str): Raw LLM response string
        Returns:
            object: Parsed Python object (list or dict)
        """
        if not chat_simple or not isinstance(chat_simple, str):
            return None

        # Extract ```json ... ``` or ``` ... ``` code blocks
        code_blocks = re.findall(r"```(?:json)?(.*?)```", chat_simple, re.DOTALL)
        if code_blocks:
            for block in code_blocks:
                try:
                    return json.loads(block.strip())
                except json.JSONDecodeError:
                    continue  # Try the next block

        # No code block? Try parsing the full string
        try:
            return json.loads(chat_simple.strip())
        except json.JSONDecodeError:
            pass

        # Try extracting the first valid JSON segment (starts with [ or {)
        json_candidates = re.findall(r"(\{.*\}|\[.*\])", chat_simple, re.DOTALL)
        for candidate in json_candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

        print("❌ Unable to parse valid JSON from LLM output")
        return None

    def display_search_results(self, results):
        """
        Display search results
        """
        print(f"\n=== Search results: '{results['query']}' ===")
        
        if results['entities']:
            print(f"\nFound {len(results['entities'])} related entities:")
            for entity in results['entities'][:5]:  # Show only the first five
                print(f"  • {entity}")
        
        if results['relationships']:
            print(f"\nFound {len(results['relationships'])} related relationships:")
            for source, target, attrs in results['relationships'][:3]:  # Show only the first three
                print(f"  • {source} --[{attrs.get('relationship', '')}]--> {target}")
        
        if results['evidence']:
            print(f"\nFound {len(results['evidence'])} related evidence snippets:")
            for source, target, attrs in results['evidence'][:3]:  # Show only the first three
                evidence = attrs.get('evidence', '')[:100]
                print(f"  • {evidence}...")
        
        if results['suggestions']:
            print(f"\nSuggested entities:")
            for suggestion in results['suggestions']:
                print(f"  • {suggestion['entity']} ({suggestion['connections']} connections)")
    
    def display_entity_info(self, info):
        """
        Display entity information
        """
        if 'error' in info:
            print(f"Error: {info['error']}")
            return
        
        print(f"\n=== Entity: {info['entity']} ===")
        print(f"Total connections: {info['total_connections']}")
        
        if info['outgoing_relationships']:
            print(f"\nOutgoing relationships ({len(info['outgoing_relationships'])}):")
            for rel in info['outgoing_relationships']:
                print(f"  → {rel['target']} ({rel['relationship']})")
                print(f"    Evidence: {rel['evidence'][:100]}...")
        
        if info['incoming_relationships']:
            print(f"\nIncoming relationships ({len(info['incoming_relationships'])}):")
            for rel in info['incoming_relationships']:
                print(f"  ← {rel['source']} ({rel['relationship']})")
                print(f"    Evidence: {rel['evidence'][:100]}...")
    
    def display_paths(self, paths):
        """
        Display paths
        """
        if not paths:
            print("No paths found")
            return
        
        print(f"\nFound {len(paths)} paths:")
        for i, path in enumerate(paths[:5]):  # Show only the first five
            print(f"  Path {i+1}: {' -> '.join(path)}")
    
    def display_related_entities(self, entity, related):
        """
        Display related entities
        """
        print(f"\n=== Related entities for {entity} ===")
        if not related:
            print("No related entities found")
            return
        
        print(f"Found {len(related)} related entities:")
        for i, rel_entity in enumerate(related[:10]):  # Show only the first ten
            print(f"  {i+1}. {rel_entity}")

    def display_search_node_and_edge(self, results):
        """
        Display combined node-edge search results
        """
        if not results:
            print("No related results found")
            return
        
        print(f"\n=== Node-edge-node search results ===")
        print(f"Found {len(results)} related results:")
        
        for i, result in enumerate(results[:10]):  # Show only the first ten
            print(f"\nResult {i+1}:")
            print(f"  Source node: {result['source_node']}")
            print(f"  Target node: {result['target_node']}")
            print(f"  Relationship: {result['edge']}")
            print(f"  Evidence: {result['evidence']}")
            print(f"  Node similarity: {result['node_similarity']:.4f}")
            print(f"  Edge similarity: {result['edge_similarity']:.4f}")
            if result['paper_id']:
                print(f"  Paper ID: {result['paper_id']}")
            if result['review_id']:
                print(f"  Review ID: {result['review_id']}")
    
    def show_stats(self):
        """
        Display graph statistics
        """
        print(f"\n=== Graph statistics ===")
        print(f"Nodes: {self.G.number_of_nodes()}")
        print(f"Edges: {self.G.number_of_edges()}")
        print(f"Weakly connected components: {nx.number_weakly_connected_components(self.G)}")
        
        # Node degree stats
        degrees = dict(self.G.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\nMost connected nodes:")
        for node, degree in top_nodes:
            print(f"  {node}: {degree} connections")

    def search_similar_node(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k nodes by semantic similarity.
        Returns: [{"node": "<name>", "similarity": <score>}, ...]
        """
        if self._entity_embeddings is None or not self._entity_names:
            return []

        # 1. Build the query vector
        try:
            q_vec = embed_texts([query], model='BAAI/bge-m3')[0]
        except Exception:
            return []

        # 2. Compute cosine similarity
        similarities = []
        query_norm = sum(a * a for a in q_vec) ** 0.5
        if query_norm == 0:
            return []

        for idx, vec in enumerate(self._entity_embeddings):
            vec_norm = sum(a * a for a in vec) ** 0.5
            if vec_norm == 0:
                similarity = 0.0
            else:
                dot_product = sum(a * b for a, b in zip(q_vec, vec))
                similarity = dot_product / (query_norm * vec_norm)

            similarities.append({
                "node": self._entity_names[idx],
                "similarity": similarity
            })

        # 3. Sort by similarity descending
        similarities.sort(key=lambda x: x["similarity"], reverse=True)

        # 4. Return the top-k results
        return similarities[:k]


    def search_similar_edge(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k edges by semantic similarity.
        Returns:
            [
                {
                    "source": "<source>",
                    "target": "<target>",
                    "relationship": "<relationship>",
                    "evidence": "<evidence>",
                    "similarity": <score>,
                    "paper_id": "<paper id>",
                    "review_id": "<review id>"
                },
                ...
            ]
        """
        if self._edge_embeddings is None:
            return []
        
        # Build the query vector
        try:
            q_vec = embed_texts([query], model='BAAI/bge-m3')[0]
        except Exception:
            return []
        
        # Compute similarities and grab the top-k
        similarities = []
        for vec in self._edge_embeddings:
            # Dot product
            dot_product = sum(a * b for a, b in zip(q_vec, vec))
            # Vector magnitude
            query_norm = sum(a * a for a in q_vec) ** 0.5
            vec_norm = sum(a * a for a in vec) ** 0.5
            
            if query_norm == 0 or vec_norm == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (query_norm * vec_norm)
            similarities.append(similarity)
        
        # Get top-k
        indexed_sims = [(i, sim) for i, sim in enumerate(similarities)]
        indexed_sims.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, similarity in indexed_sims[:k]:
            source, target, attrs = self._edge_triplets[idx]
            results.append({
                "source": source,
                "target": target,
                "relationship": attrs.get('relationship', ''),
                "evidence": attrs.get('evidence', ''),
                "similarity": similarity,
                "paper_id": attrs.get('paper_id', ''),
                "review_id": attrs.get('review_id', '')
            })
        
        return results

    def search_similar_evidence(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k evidence entries by semantic similarity.
        Returns structure identical to search_similar_edge.
        """
        if self._edge_embeddings is None:
            return []
        
        # Build the query vector
        try:
            q_vec = embed_texts([query], model='BAAI/bge-m3')[0]
        except Exception:
            return []
        
        # Compute similarities and grab the top-k
        similarities = []
        for vec in self._edge_embeddings:
            # Dot product
            dot_product = sum(a * b for a, b in zip(q_vec, vec))
            # Vector magnitude
            query_norm = sum(a * a for a in q_vec) ** 0.5
            vec_norm = sum(a * a for a in vec) ** 0.5
            
            if query_norm == 0 or vec_norm == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (query_norm * vec_norm)
            similarities.append(similarity)
        
        # Get top-k
        indexed_sims = [(i, sim) for i, sim in enumerate(similarities)]
        indexed_sims.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, similarity in indexed_sims[:k]:
            source, target, attrs = self._edge_triplets[idx]
            results.append({
                "source": source,
                "target": target,
                "relationship": attrs.get('relationship', ''),
                "evidence": attrs.get('evidence', ''),
                "similarity": similarity,
                "paper_id": attrs.get('paper_id', ''),
                "review_id": attrs.get('review_id', '')
            })
        
        return results

    def search_similar_node_and_edge(self, node_query: str, node_k: int = 5, edge_query: str = "", edge_k: int = 5) -> List[Dict[str, Any]]:
        """
        Combined node/edge retrieval:
        - Deduplicate nodes
        - Take the top node_k nodes
        - Retrieve edges for each and optionally apply edge_query filtering
        """
        # 1. Collect a large pool of candidate nodes
        similar_nodes = self.search_similar_node(node_query, node_k * edge_k * 2)
        if not similar_nodes:
            return []

        # 2. Group by node name (identical names treated as one group)
        grouped_nodes = {}
        for n in similar_nodes:
            node_name = n["node"].strip()
            sim = n["similarity"]
            if node_name not in grouped_nodes:
                grouped_nodes[node_name] = []
            grouped_nodes[node_name].append(sim)

        # 3. Compute a representative similarity per group (max value)
        node_groups = []
        for node_name, sims in grouped_nodes.items():
            rep_sim = max(sims)
            node_groups.append((node_name, rep_sim))

        # 4. Sort descending and keep the top node_k groups
        node_groups.sort(key=lambda x: x[1], reverse=True)
        top_node_groups = node_groups[:node_k]

        results = []

        # 5. Retrieve edges for each node group
        for node_name, node_similarity in top_node_groups:
            # Grab all edges touching this node
            node_edges = [
                (source, target, attrs)
                for source, target, attrs in self.G.edges(data=True)
                if source == node_name or target == node_name
            ]
            if not node_edges:
                continue

            # 6. If no edge_query is provided, take the first edge_k edges
            if not edge_query.strip():
                for source, target, attrs in node_edges[:edge_k]:
                    results.append({
                        "source_node": source,
                        "target_node": target,
                    "edge": attrs.get('relationship', ''),
                    "evidence": attrs.get('evidence', ''),
                        "paper_id": attrs.get('paper_id', ''),
                        "review_id": attrs.get('review_id', ''),
                        "node_similarity": node_similarity,
                        "edge_similarity": 1.0
                    })
                continue

            # 7. An edge_query is provided → compute semantic similarity
            if self._edge_embeddings is not None:
                try:
                    edge_q_vec = embed_texts([edge_query], model='BAAI/bge-m3')[0]
                except Exception:
                    continue

                edge_similarities = []
                for source, target, attrs in node_edges:
                    edge_text = f"EDGE: {source} --[{attrs.get('relationship', '')}]--> {target}. EVIDENCE: {attrs.get('evidence', '')}"
                    try:
                        edge_vec = embed_texts([edge_text], model='BAAI/bge-m3')[0]
                        dot_product = sum(a * b for a, b in zip(edge_q_vec, edge_vec))
                        query_norm = sum(a * a for a in edge_q_vec) ** 0.5
                        vec_norm = sum(a * a for a in edge_vec) ** 0.5
                        similarity = dot_product / (query_norm * vec_norm) if query_norm and vec_norm else 0.0
                        edge_similarities.append((source, target, attrs, similarity))
                    except Exception:
                        continue

                # Sort and take top edge_k
                edge_similarities.sort(key=lambda x: x[3], reverse=True)
                for source, target, attrs, edge_similarity in edge_similarities[:edge_k]:
                    results.append({
                        "source_node": source,
                        "target_node": target,
                        "edge": attrs.get('relationship', ''),
                        "evidence": attrs.get('evidence', ''),
                        "paper_id": attrs.get('paper_id', ''),
                        "review_id": attrs.get('review_id', ''),
                        "node_similarity": node_similarity,
                        "edge_similarity": edge_similarity
                    })
            else:
                # No edge embeddings available → take the first edge_k edges
                for source, target, attrs in node_edges[:edge_k]:
                    results.append({
                        "source_node": source,
                        "target_node": target,
                        "edge": attrs.get('relationship', ''),
                        "evidence": attrs.get('evidence', ''),
                        "paper_id": attrs.get('paper_id', ''),
                        "review_id": attrs.get('review_id', ''),
                        "node_similarity": node_similarity,
                        "edge_similarity": 1.0
                    })
        return results
        #print(results)

    def agent_retrieve_risk_points(self, query: str, top_k_nodes: int = 8, top_k_edges: int = 20, expand_hops: int = 1) -> List[Dict[str, Any]]:
        """
        Agent-facing interface for retrieving risk points:
        - Accepts a query
        - Runs vector retrieval to get top-k entities and edges
        - Expands neighbors around the top entities up to expand_hops
        - Returns a unified list of {source_node, target_node, edge, review_id, paper_id_label}

        Notes:
        - edge field uses the relationship label
        - evidence field carries the evidence text
        - review_id and paper_id (paper identifier) are included when available
        - Results are deduplicated automatically
        """
        # Run vector retrieval
        vec = self.semantic_vector_search(query, top_k=max(top_k_nodes, top_k_edges))
        top_entities: List[str] = vec.get('top_entities', []) if isinstance(vec, dict) else []
        top_edges = vec.get('top_edges', []) if isinstance(vec, dict) else []

        # Result aggregation
        collected: List[Dict[str, Any]] = []
        seen = set()

        def add_edge_record(source: str, target: str, attrs: Dict[str, Any]):
            rel = (attrs or {}).get('relationship', '')
            review_id = (attrs or {}).get('review_id', '')
            paper_id = (attrs or {}).get('paper_id', '')
            evidence = (attrs or {}).get('evidence', '')
            key = (source, target, rel, review_id, paper_id)
            if key in seen:
                return
            seen.add(key)
            collected.append({
                'source_node': source,
                'target_node': target,
                'edge': rel,
                'evidence': evidence,
                'review_id': review_id,
                'paper_id_label': paper_id,
            })

        # 1) Add the top-k edges directly
        for i, (s, t, attrs) in enumerate(top_edges[:top_k_edges]):
            add_edge_record(s, t, attrs)

        # 2) Expand neighbors for the top-k entities
        for ent in top_entities[:top_k_nodes]:
            frontier = {ent}
            visited = {ent}
            for _ in range(expand_hops):
                next_level = set()
                for node in frontier:
                    # Outgoing edges
                    for _, nb, attrs in self.G.out_edges(node, data=True):
                        add_edge_record(node, nb, attrs)
                        if nb not in visited:
                            next_level.add(nb)
                            visited.add(nb)
                    # Incoming edges
                    for nb, _, attrs in self.G.in_edges(node, data=True):
                        add_edge_record(nb, node, attrs)
                        if nb not in visited:
                            next_level.add(nb)
                            visited.add(nb)
                frontier = next_level

        return collected

def main():
    """
    Entry point
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Graph retrieval system")
    parser.add_argument("--gpu", default="auto", help="GPU device (auto, cuda:0, cuda:1, etc.)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--graph-file", default="result_v2/all_graphs_cleaned.json", help="Graph file path")
    
    args = parser.parse_args()
    
    print(f"Using GPU device: {args.gpu}")
    print(f"Batch size: {args.batch_size}")
    print(f"Graph file: {args.graph_file}")
    
    # Create the retrieval system
    retrieval_system = GraphRetrievalSystem(args.graph_file, gpu_device=args.gpu, batch_size=args.batch_size)
    
    # Launch interactive search if needed
    #retrieval_system.interactive_search()
    results = retrieval_system.search_similar_node_and_edge("Large Language Models", 5, "is better than", 5)
    #print(results)

if __name__ == "__main__":
    main()