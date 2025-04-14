import redis
import json
import hashlib
import numpy as np
import os
import time
from typing import Dict, List, Optional, Tuple, Any
from sentence_transformers import SentenceTransformer, util
from groq import Groq
from dotenv import load_dotenv
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
load_dotenv()


class LLMRedisCache:
    def __init__(
        self, 
        redis_host: str = '', 
        redis_port: int = 14799, 
        redis_db: int = 0,
        redis_username: str ="",
        redis_password: str ="",
        embedding_model: str = 'all-MiniLM-L6-v2',
        groq_api_key: str = None,
        llm_model: str = "llama-3.3-70b-versatile",
        similarity_threshold: float = 0.85,
        ttl: int = 86400 * 7,  # Cache TTL (7 days default)
        index_name: str = "llm_embeddings_idx"
    ):
        """
        Initialize Redis connection, embedding model, and Groq API settings.
        
        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
            embedding_model: SentenceTransformer model name
            groq_api_key: API key for Groq (defaults to GROQ_API_KEY environment variable)
            similarity_threshold: Threshold for semantic similarity (0-1)
            llm_model: Groq model name to use
            ttl: Time-to-live for cache entries in seconds
            index_name: Name of the Redis vector search index
        """
        # Initialize Redis clients
        self.redis_client = redis.Redis(
            host=redis_host, 
            port=redis_port, 
            db=redis_db,
            decode_responses=True,
            username=redis_username,
            password=redis_password
        )
        
        # Initialize embedding model
        self.model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Groq API settings
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("Groq API key is required either as parameter or GROQ_API_KEY environment variable")
        
        # Initialize Groq client
        self.groq_client = Groq(api_key=self.groq_api_key)
        self.llm_model = llm_model
        
        self.similarity_threshold = similarity_threshold
        self.ttl = ttl
        
        # Key prefixes for different cache types
        self.keyword_prefix = "llm:keyword:"
        self.semantic_prefix = "llm:semantic:"
        self.vector_prefix = "llm:vector:"
        self.metadata_prefix = "llm:metadata:"
        
        # Vector search index name
        self.index_name = index_name
        
        # Create vector search index if it doesn't exist
        self._create_vector_index()
    
    def _create_vector_index(self):
        """Create a Redis vector search index for efficient similarity search."""
        try:
            # Check if index already exists
            self.redis_client.ft(self.index_name).info()
            print(f"Vector index '{self.index_name}' already exists")
        except:
            # Create the index
            schema = [
                TextField("query"),
                TextField("response_key"),
                VectorField(
                    "embedding", 
                    "FLAT", 
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.embedding_dim,
                        "DISTANCE_METRIC": "COSINE",
                    }
                )
            ]
            
            # Create the index with the schema
            self.redis_client.ft(self.index_name).create_index(
                schema, 
                definition=IndexDefinition(
                    prefix=[self.vector_prefix],
                    index_type=IndexType.HASH
                )
            )
            print(f"Created vector index '{self.index_name}'")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generate embeddings using SentenceTransformers.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array of embeddings
        """
        start_time = time.time()
        # Generate embedding and normalize
        embedding = self.model.encode(text)
        elapsed_time = time.time() - start_time
        print(f"Embedding generation time: {elapsed_time:.4f} seconds")
        return embedding
    
    def get_llm_response(self, query: str, system_prompt: str = "You are a helpful assistant.", **params) -> Dict[str, Any]:
        """
        Get response from Groq API.
        
        Args:
            query: The query to send to Groq
            system_prompt: System prompt to use
            **params: Additional parameters for the API
            
        Returns:
            Response from Groq API
        """
        start_time = time.time()
        
        # Default parameters
        default_params = {
            "temperature": 0.7,
            "max_tokens": 500,
            "top_p": 1.0,
        }
        
        # Update with user-provided parameters
        api_params = default_params.copy()
        api_params.update(params)
        
        # Define messages
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": query
            }
        ]
        
        # Make API call to Groq
        response = self.groq_client.chat.completions.create(
            messages=messages,
            model=self.llm_model,
            **api_params
        )
        
        elapsed_time = time.time() - start_time
        
        # Format the response consistently for caching
        formatted_response = {
            "prompt": query,
            "system_prompt": system_prompt,
            "response_text": response.choices[0].message.content,
            "response_object": {
                "id": response.id,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            },
            "timestamp": time.time(),
            "response_time": elapsed_time
        }
        
        print(f"LLM API call time: {elapsed_time:.4f} seconds")
        return formatted_response
    
    def _generate_hash(self, query: str, system_prompt: str = "") -> str:
        """Generate a deterministic hash for a query and system prompt combination."""
        combined = f"{query}|{system_prompt}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _store_in_cache(self, key: str, response: Dict[str, Any], embedding: Optional[np.ndarray] = None) -> None:
        """Store response in cache with TTL."""
        start_time = time.time()
        
        # Store the response
        self.redis_client.set(key, json.dumps(response))
        self.redis_client.expire(key, self.ttl)
        
        # If embedding is provided, store it for vector search
        if embedding is not None:
            # Generate a hash key for the vector entry
            hash_key = self._generate_hash(key)
            vector_key = f"{self.vector_prefix}{hash_key}"
            
            # Prepare vector data for storage
            vector_data = {
                "query": key.replace(self.semantic_prefix, ""),
                "response_key": key,
                "embedding": embedding.astype(np.float32).tobytes()
            }
            
            # Store vector data in Redis hash
            self.redis_client.hset(vector_key, mapping={
                "query": vector_data["query"],
                "response_key": vector_data["response_key"],
                "embedding": vector_data["embedding"]
            })
            self.redis_client.expire(vector_key, self.ttl)
            
            # Store metadata separately for easier access
            metadata = {
                "query": key.replace(self.semantic_prefix, ""),
                "timestamp": time.time(),
                "response_key": key
            }
            metadata_key = f"{self.metadata_prefix}{hash_key}"
            self.redis_client.set(metadata_key, json.dumps(metadata))
            self.redis_client.expire(metadata_key, self.ttl)
        
        elapsed_time = time.time() - start_time
        print(f"Cache storage time: {elapsed_time:.4f} seconds")
    
    def _get_from_keyword_cache(self, query: str, system_prompt: str) -> Optional[Dict[str, Any]]:
        """Try to get response from keyword cache."""
        start_time = time.time()
        
        hash_key = self._generate_hash(query, system_prompt)
        key = f"{self.keyword_prefix}{hash_key}"
        cached_response = self.redis_client.get(key)
        
        elapsed_time = time.time() - start_time
        
        if cached_response:
            print(f"Keyword cache hit! Retrieval time: {elapsed_time:.4f} seconds")
            return json.loads(cached_response)
        
        print(f"Keyword cache miss. Check time: {elapsed_time:.4f} seconds")
        return None
    
    def _find_similar_query(self, query_embedding: np.ndarray) -> Optional[Tuple[str, float]]:
        """
        Find semantically similar query in cache using Redis vector search.
        Returns tuple of (cache_key, similarity_score) if found, None otherwise.
        """
        start_time = time.time()
        
        # Prepare the query embedding for Redis vector search
        query_vector = query_embedding.astype(np.float32).tobytes()
        
        # Perform vector search using Redis
        search_query = f"*=>[KNN 1 @embedding $query_vector AS score]"
        query_params = {
            "query_vector": query_vector
        }
        
        # Run the search
        try:
            results = self.redis_client.ft(self.index_name).search(
                search_query,
                query_params=query_params
            )
            
            # Check if we have results
            if results.total > 0 and hasattr(results, 'docs') and len(results.docs) > 0:
                # Get the top result
                top_result = results.docs[0]
                similarity = 1 - float(top_result.score)  # Convert distance to similarity
                
                if similarity >= self.similarity_threshold:
                    response_key = top_result.response_key
                    elapsed_time = time.time() - start_time
                    print(f"Vector search time: {elapsed_time:.4f} seconds")
                    print(f"Found similar query with similarity: {similarity:.4f}")
                    return (response_key, similarity)
        
        except Exception as e:
            print(f"Vector search error: {e}")
        
        elapsed_time = time.time() - start_time
        print(f"Vector search time: {elapsed_time:.4f} seconds")
        print("No similar queries found above threshold")
        return None
    
    def _get_from_semantic_cache(self, query_embedding: np.ndarray) -> Optional[Dict[str, Any]]:
        """Try to get response from semantic cache using vector search."""
        similar_query = self._find_similar_query(query_embedding)
        
        if similar_query:
            start_time = time.time()
            
            cache_key, similarity = similar_query
            cached_response = self.redis_client.get(cache_key)
            
            elapsed_time = time.time() - start_time
            
            if cached_response:
                print(f"Semantic cache hit! Retrieval time: {elapsed_time:.4f} seconds")
                return json.loads(cached_response)
        
        return None
    
    def get_response(self, query: str, system_prompt: str = "You are a helpful assistant.", **llm_params) -> Dict[str, Any]:
        """
        Get response for query, using caching strategy:
        1. Try keyword cache first
        2. If not found, try semantic cache
        3. If still not found, call Groq API
        
        Args:
            query: The query to process
            system_prompt: System prompt to use
            **llm_params: Additional parameters to pass to the LLM function
            
        Returns:
            The LLM response dictionary
        """
        total_start_time = time.time()
        
        # 1. Try keyword cache
        keyword_response = self._get_from_keyword_cache(query, system_prompt)
        if keyword_response:
            total_elapsed = time.time() - total_start_time
            print(f"Total response time (keyword cache): {total_elapsed:.4f} seconds")
            
            # Add timing info to the response
            keyword_response["response_source"] = "keyword_cache"
            keyword_response["total_response_time"] = total_elapsed
            return keyword_response
            
        # 2. Get embedding for semantic search
        query_embedding = self.get_embedding(query)
        
        # 3. Try semantic cache
        semantic_response = self._get_from_semantic_cache(query_embedding)
        if semantic_response:
            total_elapsed = time.time() - total_start_time
            print(f"Total response time (semantic cache): {total_elapsed:.4f} seconds")
            
            # Add timing info to the response
            semantic_response["response_source"] = "semantic_cache"
            semantic_response["total_response_time"] = total_elapsed
            return semantic_response
            
        # 4. Call Groq API as last resort
        print("Cache miss - calling Groq API")
        llm_response = self.get_llm_response(query, system_prompt, **llm_params)
        
        # 5. Store in both caches for future use
        hash_key = self._generate_hash(query, system_prompt)
        keyword_key = f"{self.keyword_prefix}{hash_key}"
        self._store_in_cache(keyword_key, llm_response)
        
        semantic_key = f"{self.semantic_prefix}{hash_key}"
        self._store_in_cache(semantic_key, llm_response, query_embedding)
        
        total_elapsed = time.time() - total_start_time
        print(f"Total response time (API call): {total_elapsed:.4f} seconds")
        
        # Add timing info to the response
        llm_response["response_source"] = "api_call"
        llm_response["total_response_time"] = total_elapsed
        return llm_response
    
    def print_timing_summary(self, response: Dict[str, Any]) -> None:
        """Print a summary of timing metrics from the response."""
        print("\n===== Timing Summary =====")
        print(f"Response source: {response.get('response_source', 'unknown')}")
        print(f"Total response time: {response.get('total_response_time', 0):.4f} seconds")
        
        if response.get('response_source') == "api_call":
            print(f"API call time: {response.get('response_time', 0):.4f} seconds")
            print(f"Tokens: {response.get('response_object', {}).get('usage', {}).get('total_tokens', 0)}")
        
        print("==========================\n")

# Example usage:
if __name__ == "__main__":
    # Example usage with Groq API
    cache = LLMRedisCache(
        redis_host = '', 
        redis_port = , 
        redis_db = 0,
        redis_username ="default",
        redis_password ="",
        # If GROQ_API_KEY is set in environment, this is optional
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        embedding_model="all-MiniLM-L6-v2",
        llm_model="llama-3.3-70b-versatile",
        similarity_threshold=0.50
    )
    
    # Example query
    query = "How is AI commonly used in our everyday activities?"
    
    # This call will check cache first, then call Groq if needed
    response = cache.get_response(
        query,
        system_prompt="You are a helpful assistant.",
        temperature=0.7,
        max_tokens=500
    )
    
    # Print timing summary
    cache.print_timing_summary(response)
    
    print(f"Response: {response['response_text']}")
    
    print("\n--- Second query (should hit keyword cache) ---")
    # Second call with same query should hit the keyword cache
    response2 = cache.get_response(query)
    cache.print_timing_summary(response2)
    
    print("\n--- Similar query (should hit semantic cache) ---")
    # Similar query should hit semantic cache
    similar_query = "Why are fast LLMs important?"
    response3 = cache.get_response(similar_query)
    cache.print_timing_summary(response3)