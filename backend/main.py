from math import dist
import os
import hashlib
import logging
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
from datetime import datetime
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
app.config["MONGO_URI"] = os.getenv("MONGO_URI", "mongodb://localhost:27017/rag_db")
mongo = PyMongo(app)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Langchain with Gemini
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4)

# Text splitter for chunking content
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,  # Slightly smaller for better precision
    chunk_overlap=150,
    length_function=len,
    separators=["\n\n", "\n", ".", "!", "?", ";", ",", " ", ""]  # Better semantic splitting
)

# Define prompt for date extraction
date_extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at extracting date ranges from text. Extract both start and end dates mentioned in the user's query.

If relative dates are mentioned (e.g., 'yesterday', 'last week', 'next month'), convert them to absolute dates based on the current date.

If no dates are explicitly mentioned, return:
start_date: None
end_date: None

If only one date is mentioned, use it as both start and end date.

The current date is {current_date}.
Format dates as YYYY-MM-DD.

Return the dates in the exact JSON format:
{{
  "start_date": "YYYY-MM-DD",
   "end_date": "YYYY-MM-DD"
}}
"""),
    ("user", "{query}")
])

# Add relevance checking prompt
relevance_check_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at determining if a piece of content is relevant to answer a specific question.

Analyze the content and determine if it contains information that could help answer the question.

Return a JSON response with:
- "is_relevant": true/false
- "relevance_score": a number between 0-1 (1 being most relevant)
- "reason": brief explanation of why it's relevant or not

Be strict - only mark as relevant if the content actually contains information that helps answer the question.
"""),
    ("user", "Question: {question}\n\nContent: {content}")
])

date_extraction_parser = JsonOutputParser()
date_extraction_chain = date_extraction_prompt | llm | date_extraction_parser

relevance_parser = JsonOutputParser()
relevance_chain = relevance_check_prompt | llm | relevance_parser

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="rag_collection")

def extract_keywords(text: str, top_k: int = 10) -> list:
    """
    Extract keywords from text using simple frequency analysis
    """
    # Simple keyword extraction - remove common words and get frequent terms
    stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by', 'that', 'this', 'it', 'from', 'be', 'are', 'was', 'were', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must'}
    
    # Clean and tokenize
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    filtered_words = [word for word in words if word not in stop_words]
    
    # Get most frequent words
    word_freq = Counter(filtered_words)
    return [word for word, _ in word_freq.most_common(top_k)]

def calculate_keyword_overlap(query_keywords: list, content_keywords: list) -> float:
    """
    Calculate keyword overlap between query and content
    """
    if not query_keywords or not content_keywords:
        return 0.0
    
    query_set = set(query_keywords)
    content_set = set(content_keywords)
    
    intersection = len(query_set.intersection(content_set))
    union = len(query_set.union(content_set))
    
    return intersection / union if union > 0 else 0.0

def quick_relevance_filter(question: str, content: str, min_score: float = 0.1) -> dict:
    """
    Fast relevance filtering using keyword overlap and basic heuristics
    """
    question_keywords = extract_keywords(question, top_k=5)
    content_keywords = extract_keywords(content, top_k=15)
    
    # Calculate keyword overlap
    keyword_score = calculate_keyword_overlap(question_keywords, content_keywords)
    
    # Additional heuristics
    question_lower = question.lower()
    content_lower = content.lower()
    
    # Check for exact phrase matches
    question_words = question_lower.split()
    phrase_matches = 0
    for i in range(len(question_words) - 1):
        phrase = f"{question_words[i]} {question_words[i+1]}"
        if phrase in content_lower:
            phrase_matches += 1
    
    phrase_score = phrase_matches / max(len(question_words) - 1, 1)
    
    # Combine scores
    final_score = (keyword_score * 0.7) + (phrase_score * 0.3)
    
    return {
        'is_relevant': final_score >= min_score,
        'relevance_score': final_score,
        'keyword_score': keyword_score,
        'phrase_score': phrase_score
    }

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="rag_collection")


@app.route('/')
def home():
    return "RAG Backend is running!"

def generate_url_hash(url: str) -> str:
    """
    Generates a SHA256 hash for a given URL.
    """
    return hashlib.sha256(url.encode('utf-8')).hexdigest()

def generate_chunk_id(url_hash: str, chunk_index: int) -> str:
    """
    Generate a unique ID for each chunk
    """
    return f"{url_hash}_chunk_{chunk_index}"

@app.route('/save_page', methods=['POST'])
def save_page():
    # This endpoint is called from background.js to save page content.
    # It reuses the logic from process_content.
    return process_content()

@app.route('/process_content', methods=['POST'])
def process_content():
    data = request.json
    url = data.get('url')
    content = data.get('content')
    title = data.get('title')
    favicon_url = data.get('faviconUrl')

    logger.info(f"Received request to process content for URL: {url}")
    if not url or not content:
        logger.warning("Missing URL or content in process_content request.")
        return jsonify({"error": "Missing url or content"}), 400

    try:
        url_hash = generate_url_hash(url)
        logger.info(f"Generated URL hash: {url_hash}")

        # Store metadata in MongoDB with upsert to handle duplicates
        page_metadata = {
            "_id": url_hash, # Use hash as _id
            "url": url,
            "content": content,
            "title": title,
            "favicon_url": favicon_url,
            "timestamp": datetime.utcnow()
        }
        mongo.db.pages.update_one(
            {"_id": url_hash},
            {"$set": page_metadata},
            upsert=True
        )
        logger.info(f"Content metadata stored/updated in MongoDB for URL hash: {url_hash}")

        # Delete existing chunks for this URL to avoid duplicates
        try:
            existing_chunks = collection.get(where={"url": url})
            if existing_chunks['ids']:
                collection.delete(ids=existing_chunks['ids'])
                logger.info(f"Deleted {len(existing_chunks['ids'])} existing chunks for URL: {url}")
        except Exception as e:
            logger.warning(f"Error deleting existing chunks: {e}")

        # Clean and preprocess content
        cleaned_content = clean_content(content)
        
        # Split content into chunks
        chunks = text_splitter.split_text(cleaned_content)
        logger.info(f"Split content into {len(chunks)} chunks")

        # Filter out very short or low-quality chunks
        filtered_chunks = []
        for i, chunk in enumerate(chunks):
            # Skip very short chunks (less than 50 characters)
            if len(chunk.strip()) < 50:
                logger.debug(f"Skipping short chunk {i}: {len(chunk)} characters")
                continue
            
            # Skip chunks that are mostly non-alphanumeric
            alphanumeric_ratio = sum(c.isalnum() or c.isspace() for c in chunk) / len(chunk)
            if alphanumeric_ratio < 0.6:
                logger.debug(f"Skipping low-quality chunk {i}: {alphanumeric_ratio:.2f} alphanumeric ratio")
                continue
                
            filtered_chunks.append((i, chunk))
        
        logger.info(f"Filtered to {len(filtered_chunks)} quality chunks from {len(chunks)} total")

        if not filtered_chunks:
            logger.warning(f"No quality chunks found for URL: {url}")
            return jsonify({"error": "No quality content chunks found"}), 400

        # Generate embeddings and store in ChromaDB
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Process chunks in batches to avoid memory issues
        batch_size = 10
        total_chunks_added = 0
        
        for i in range(0, len(filtered_chunks), batch_size):
            batch_data = filtered_chunks[i:i + batch_size]
            batch_chunks = [chunk for _, chunk in batch_data]
            batch_original_indices = [orig_idx for orig_idx, _ in batch_data]
            
            try:
                # Generate embeddings for this batch
                batch_embeddings = embedding_model.embed_documents(batch_chunks)
                
                # Validate embeddings
                if not batch_embeddings or len(batch_embeddings) != len(batch_chunks):
                    logger.error(f"Embedding generation failed for batch {i//batch_size + 1}")
                    continue
                
                # Generate IDs and metadata
                batch_ids = [generate_chunk_id(url_hash, orig_idx) for orig_idx in batch_original_indices]
                batch_metadatas = []
                
                for j, orig_idx in enumerate(batch_original_indices):
                    # Extract keywords for each chunk
                    chunk_keywords = extract_keywords(batch_chunks[j], top_k=10)
                    
                    metadata = {
                        "url": url,
                        "timestamp": page_metadata["timestamp"].isoformat(),
                        "title": title if title is not None else "",
                        "favicon_url": favicon_url if favicon_url is not None else "",
                        "chunk_index": orig_idx,
                        "url_hash": url_hash,
                        "keywords": " ".join(chunk_keywords),  # Store keywords for faster filtering
                        "chunk_length": len(batch_chunks[j]),
                        "chunk_hash": hashlib.md5(batch_chunks[j].encode()).hexdigest()[:16]  # For deduplication
                    }
                    batch_metadatas.append(metadata)
                
                # Add to ChromaDB
                collection.add(
                    documents=batch_chunks,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                
                total_chunks_added += len(batch_chunks)
                logger.info(f"Added batch {i//batch_size + 1} of {len(batch_chunks)} chunks to ChromaDB")
                
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                continue

        if total_chunks_added == 0:
            logger.error(f"Failed to add any chunks for URL: {url}")
            return jsonify({"error": "Failed to process any content chunks"}), 500

        logger.info(f"Successfully processed {total_chunks_added} chunks for URL: {url}")
        
        # Optional: Verify the data was stored correctly
        try:
            verification = collection.get(where={"url": url}, limit=1)
            if not verification['ids']:
                logger.warning(f"Verification failed: No chunks found in ChromaDB for URL: {url}")
        except Exception as e:
            logger.warning(f"Error during verification: {e}")
        
    except Exception as e:
        logger.error(f"Error processing content for URL {url}: {e}")
        return jsonify({"error": f"Failed to process content: {e}"}), 500

    return jsonify({
        "message": "Content processed successfully", 
        "url": url, 
        "url_hash": url_hash, 
        "chunks_created": total_chunks_added,
        "chunks_filtered": len(chunks) - len(filtered_chunks)
    }), 200


def clean_content(content):
    """Clean and preprocess content before chunking."""
    if not content:
        return ""
    
    # Remove excessive whitespace
    content = re.sub(r'\s+', ' ', content)
    
    # Remove common web artifacts
    content = re.sub(r'<[^>]+>', '', content)  # Remove HTML tags
    content = re.sub(r'&[a-zA-Z]+;', ' ', content)  # Remove HTML entities
    content = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', content)  # Remove URLs
    
    # Remove excessive punctuation
    content = re.sub(r'[^\w\s.,!?;:()\-\'"]+', ' ', content)
    
    # Remove lines that are mostly navigation or boilerplate
    lines = content.split('\n')
    filtered_lines = []
    for line in lines:
        line = line.strip()
        if len(line) < 10:  # Skip very short lines
            continue
        if any(keyword in line.lower() for keyword in ['cookie', 'privacy policy', 'terms of service', 'subscribe', 'newsletter']):
            continue
        filtered_lines.append(line)
    
    return '\n'.join(filtered_lines).strip()


def extract_keywords(text, top_k=10):
    """Extract keywords from text for faster filtering."""
    if not text:
        return []
    
    # Simple keyword extraction - you can replace with more sophisticated methods
    import re
    from collections import Counter
    
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
    
    # Extract words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Filter out stop words and get most common
    filtered_words = [word for word in words if word not in stop_words]
    word_counts = Counter(filtered_words)
    
    return [word for word, _ in word_counts.most_common(top_k)]

@app.route('/query_pages', methods=['POST'])
def query_pages():
    data = request.json
    question = data.get('question')
    similarity_threshold = data.get('similarity_threshold', 0.5)  # Lowered threshold
    max_results = data.get('max_results', 8)  # Reduced max results
    use_llm_rerank = data.get('use_llm_rerank', False)  # Optional LLM re-ranking

    logger.info(f"Received query for question: {question}")
    if not question:
        logger.warning("Missing question in query_content request.")
        return jsonify({"error": "Missing question"}), 400

    try:
        # Extract date using Langchain
        current_date = datetime.now().strftime("%Y-%m-%d")
        extracted_date_json = date_extraction_chain.invoke({"query": question, "current_date": current_date})
        start_date_str = extracted_date_json.get('start_date')
        end_date_str = extracted_date_json.get('end_date')
        logger.info(f"Extracted dates for query '{question}': start_date={start_date_str}, end_date={end_date_str}")

        # Generate embedding for the question
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        question_embedding = embedding_model.embed_query(question)
        logger.info("Generated embedding for the question.")

        # Extract keywords from question for fast filtering
        question_keywords = extract_keywords(question, top_k=5)
        logger.info(f"Extracted keywords from question: {question_keywords}")

        # Query ChromaDB with higher initial results for better filtering
        chroma_query_results = collection.query(
            query_embeddings=[question_embedding],
            n_results=min(max_results * 3, 30),  # Get more results for better filtering
            include=['metadatas', 'documents', 'distances']  # Include distances instead of embeddings
        )
        logger.info(f"ChromaDB query returned {len(chroma_query_results['ids'][0]) if chroma_query_results['ids'] else 0} results.")

        if not chroma_query_results['ids'] or not chroma_query_results['ids'][0]:
            logger.info("No results found in ChromaDB.")
            return jsonify({"message": "No relevant content found for your query.", "urls": []}), 200

        # Fast filtering with multiple criteria
        relevant_chunks = []
        
        for i, doc_id in enumerate(chroma_query_results['ids'][0]):
            metadata = chroma_query_results['metadatas'][0][i]
            content = chroma_query_results['documents'][0][i]
            
            # Get distance from ChromaDB
            distance = chroma_query_results['distances'][0][i] if 'distances' in chroma_query_results else 0.5
            
            # Convert distance to similarity score
            # For cosine distance (most common): similarity = 1 - distance
            # For euclidean distance: similarity = 1 / (1 + distance)
            # Check your ChromaDB distance metric configuration
            
            # Assuming cosine distance (default for most embeddings)
            similarity_score = 1 - distance
            logger.info(f"Distance of {i} for chunk {doc_id} (cosine distance): {distance:.3f}, Calculated similarity score: {similarity_score:.3f}")
            
            # Alternative for euclidean distance (uncomment if using euclidean):
            # similarity_score = 1 / (1 + distance)
            
            logger.info(f"Distance of {i} for chunk {doc_id}: {distance:.3f}, Calculated similarity score: {similarity_score:.3f}")
            
            # Apply similarity threshold (first filter)
            if similarity_score < similarity_threshold:
                logger.debug(f"Skipping chunk {doc_id} due to low similarity score: {similarity_score:.3f}")
                continue

            # Apply date filter if provided (second filter)
            if start_date_str and end_date_str:
                try:
                    doc_date = datetime.fromisoformat(metadata['timestamp']).date()
                    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
                    end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()

                    if not (start_date <= doc_date <= end_date):
                        logger.debug(f"Skipping chunk {doc_id} due to date filter mismatch.")
                        continue
                except ValueError:
                    logger.warning(f"Could not parse extracted date(s): start_date={start_date_str}, end_date={end_date_str}")

            # Fast keyword-based relevance check (third filter)
            content_keywords = metadata.get('keywords', '').split() if metadata.get('keywords') else extract_keywords(content, top_k=10)
            keyword_overlap = calculate_keyword_overlap(question_keywords, content_keywords)
            
            # Quick heuristic relevance check
            quick_relevance = quick_relevance_filter(question, content, min_score=0.1)
            
            # Combine scores for ranking
            combined_score = (
                similarity_score * 0.4 +
                quick_relevance['relevance_score'] * 0.4 +
                keyword_overlap * 0.2
            )
            
            # Only keep if it passes basic relevance
            if quick_relevance['is_relevant'] or similarity_score > 0.8:
                relevant_chunks.append({
                    'content': content,
                    'metadata': metadata,
                    'similarity_score': similarity_score,
                    'keyword_overlap': keyword_overlap,
                    'quick_relevance_score': quick_relevance['relevance_score'],
                    'combined_score': combined_score
                })
                logger.debug(f"Chunk {doc_id} passed filtering with combined score: {combined_score:.3f}")

        # Sort by combined score
        relevant_chunks.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Take top results and optionally apply LLM re-ranking to top candidates
        if use_llm_rerank and len(relevant_chunks) > max_results:
            # Only use LLM re-ranking on top candidates to reduce API calls
            top_candidates = relevant_chunks[:max_results * 2]
            
            # Batch LLM relevance check (more efficient)
            llm_checked_chunks = []
            batch_content = []
            batch_indices = []
            
            for idx, chunk in enumerate(top_candidates):
                batch_content.append(chunk['content'][:400])  # Truncate for faster processing
                batch_indices.append(idx)
                
                # Process in batches of 5 to avoid rate limits
                if len(batch_content) == 5 or idx == len(top_candidates) - 1:
                    try:
                        # Single API call for multiple chunks
                        batch_prompt = f"""Rate the relevance of each content snippet to the question: {question}

Rate each snippet from 0-1 (1 being most relevant). Only return the scores separated by commas.

Content snippets:
""" + "\n\n".join([f"{i+1}. {content}" for i, content in enumerate(batch_content)])
                        
                        model = genai.GenerativeModel('gemini-2.0-flash')
                        response = model.generate_content(batch_prompt)
                        scores = [float(s.strip()) for s in response.text.split(',')]
                        
                        # Apply LLM scores
                        for batch_idx, score in enumerate(scores):
                            original_idx = batch_indices[batch_idx]
                            chunk = top_candidates[original_idx]
                            chunk['llm_relevance_score'] = score
                            chunk['final_score'] = chunk['combined_score'] * 0.6 + score * 0.4
                            
                            if score > 0.3:  # Only keep if LLM thinks it's relevant
                                llm_checked_chunks.append(chunk)
                                
                    except Exception as e:
                        logger.warning(f"Error in batch LLM relevance check: {e}")
                        # Fallback to original chunks
                        for batch_idx in batch_indices:
                            chunk = top_candidates[batch_idx]
                            chunk['llm_relevance_score'] = chunk['combined_score']
                            chunk['final_score'] = chunk['combined_score']
                            llm_checked_chunks.append(chunk)
                    
                    # Reset batch
                    batch_content = []
                    batch_indices = []
            
            # Re-sort by final score
            llm_checked_chunks.sort(key=lambda x: x['final_score'], reverse=True)
            relevant_chunks = llm_checked_chunks[:max_results]
        else:
            relevant_chunks = relevant_chunks[:max_results]
        
        logger.info(f"Found {len(relevant_chunks)} relevant chunks after filtering.")

        if not relevant_chunks:
            logger.info("No relevant content found after processing query and filters.")
            return jsonify({"message": "No relevant content found for your query.", "urls": []}), 200

        # Group chunks by URL to avoid duplicate URLs
        url_groups = {}
        for chunk in relevant_chunks:
            url = chunk['metadata']['url']
            if url not in url_groups:
                url_groups[url] = {
                    'url': url,
                    'title': chunk['metadata'].get('title', ''),
                    'favicon_url': chunk['metadata'].get('favicon_url', ''),
                    'chunks': []
                }
            url_groups[url]['chunks'].append(chunk)

        # Create URL list with best chunks per URL
        relevant_urls = []
        for url, group in url_groups.items():
            # Sort chunks by final score for this URL
            group['chunks'].sort(key=lambda x: x.get('final_score', x['combined_score']), reverse=True)
            best_score = group['chunks'][0].get('final_score', group['chunks'][0]['combined_score'])
            relevant_urls.append({
                'url': url,
                'title': group['title'],
                'favicon_url': group['favicon_url'],
                'relevance_score': best_score
            })

        # Sort URLs by relevance
        relevant_urls.sort(key=lambda x: x['relevance_score'], reverse=True)

        # Use Gemini for final answer generation with optimized prompt
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Create concise context from top chunks
        context_parts = []
        for chunk in relevant_chunks[:5]:  # Only use top 5 chunks
            context_parts.append(f"Source: {chunk['metadata']['url']}\n{chunk['content'][:300]}...")  # Truncate content
        
        relevant_contents_str = '\n\n'.join(context_parts)
        
        # More efficient prompt
        prompt = f"""Answer this question based on the provided context: {question}

Context:
{relevant_contents_str}

Provide a concise, accurate answer. If you reference specific information, mention the source URL.

Question: {question}"""
        
        logger.info("Sending optimized prompt to Gemini model.")
        response = model.generate_content(prompt)
        logger.info("Received response from Gemini model.")

        # Clean up URL format for frontend
        source_urls_for_frontend = [
            {
                "url": url_obj['url'],
                "favicon_url": url_obj['favicon_url'],
                "title": url_obj['title']
            }
            for url_obj in relevant_urls
        ]

        result = {
            "answer": response.text,
            "source_urls": source_urls_for_frontend,
            "total_chunks_found": len(relevant_chunks),
            "similarity_threshold_used": similarity_threshold,
            "llm_rerank_used": use_llm_rerank
        }
        
        logger.info(f"Returning result with {len(source_urls_for_frontend)} unique URLs")
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error querying content for question '{question}': {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_saved_pages', methods=['GET'])
def get_saved_pages():
    try:
        # Fetch all pages from MongoDB
        pages_cursor = mongo.db.pages.find({}).sort("timestamp", -1)
        saved_pages = []
        for page in pages_cursor:
            saved_pages.append({
                "url": page["url"],
                "title": page.get("title", "No Title"),
                "favicon_url": page.get("favicon_url", ""),
                "date": page["timestamp"].isoformat()
            })
        logger.info(f"Retrieved {len(saved_pages)} saved pages from MongoDB.")
        return jsonify(saved_pages), 200
    except Exception as e:
        logger.error(f"Error retrieving saved pages: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/delete_page', methods=['POST'])
def delete_page():
    data = request.json
    url = data.get('url')

    if not url:
        logger.warning("Missing URL in delete_page request.")
        return jsonify({"error": "Missing URL"}), 400

    try:
        url_hash = generate_url_hash(url)

        # Delete from MongoDB
        mongo.db.pages.delete_one({"_id": url_hash})
        logger.info(f"Deleted page with URL hash {url_hash} from MongoDB.")

        # Delete all chunks for this URL from ChromaDB
        chunks_to_delete = collection.get(where={"url": url})
        if chunks_to_delete['ids']:
            collection.delete(ids=chunks_to_delete['ids'])
            logger.info(f"Deleted {len(chunks_to_delete['ids'])} chunks for URL {url} from ChromaDB.")

        return jsonify({"message": "Page deleted successfully"}), 200
    except Exception as e:
        logger.error(f"Error deleting page with URL {url}: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)