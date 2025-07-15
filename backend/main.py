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

        # Split content into chunks
        chunks = text_splitter.split_text(content)
        logger.info(f"Split content into {len(chunks)} chunks")

        # Generate embeddings and store in ChromaDB
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Process chunks in batches to avoid memory issues
        batch_size = 10
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = embedding_model.embed_documents(batch_chunks)
            
            batch_ids = [generate_chunk_id(url_hash, i + j) for j in range(len(batch_chunks))]
            batch_metadatas = [{
                "url": url,
                "timestamp": page_metadata["timestamp"].isoformat(),
                "title": title if title is not None else "",
                "favicon_url": favicon_url if favicon_url is not None else "",
                "chunk_index": i + j,
                "url_hash": url_hash
            } for j in range(len(batch_chunks))]
            
            collection.add(
                documents=batch_chunks,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            logger.info(f"Added batch {i//batch_size + 1} of chunks to ChromaDB")

        logger.info(f"Successfully processed {len(chunks)} chunks for URL: {url}")
        
    except Exception as e:
        logging.error(f"Error processing content for URL {url}: {e}")
        return jsonify({"error": f"Failed to process content: {e}"}), 500

    return jsonify({"message": "Content processed successfully", "url": url, "url_hash": url_hash, "chunks_created": len(chunks)}), 200

@app.route('/query_pages', methods=['POST'])
def query_pages():
    data = request.json
    question = data.get('question')
    similarity_threshold = data.get('similarity_threshold', 0.6)  # Lowered threshold
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

        # Query ChromaDB for relevant content with more results for filtering
        chroma_query_results = collection.query(
            query_embeddings=[question_embedding],
            n_results=min(max_results * 2, 20),  # Get more results for better filtering
            include=['embeddings', 'metadatas', 'documents']  # Explicitly request embeddings
        )
        logger.info(f"ChromaDB query returned {len(chroma_query_results['ids'][0]) if chroma_query_results['ids'] else 0} results.")

        if not chroma_query_results['ids'] or not chroma_query_results['ids'][0]:
            logger.info("No results found in ChromaDB.")
            return jsonify({"message": "No relevant content found for your query.", "urls": []}), 200

        # Calculate similarity scores and filter
        question_embedding_np = np.array(question_embedding).reshape(1, -1)
        relevant_chunks = []
        
        for i, doc_id in enumerate(chroma_query_results['ids'][0]):
            metadata = chroma_query_results['metadatas'][0][i]
            content = chroma_query_results['documents'][0][i]
            
            # Calculate cosine similarity - check if embeddings are available
            similarity_score = 0.5  # Default similarity score
            if ('embeddings' in chroma_query_results and 
                chroma_query_results['embeddings'] and 
                len(chroma_query_results['embeddings'][0]) > i):
                try:
                    logger.info(f"Calculating similarity score for chunk {doc_id}")
                    chunk_embedding = np.array(chroma_query_results['embeddings'][0][i]).reshape(1, -1)
                    similarity_score = cosine_similarity(question_embedding_np, chunk_embedding)[0][0]
                    logger.info(f"Successfully calculated similarity score: {similarity_score}")
                except Exception as e:
                    logger.warning(f"Error calculating similarity for chunk {doc_id}: {e}")
                    # Fallback: use distance-based similarity if available
                    if ('distances' in chroma_query_results and 
                        chroma_query_results['distances'] and 
                        len(chroma_query_results['distances'][0]) > i):
                        # ChromaDB distance is typically cosine distance, convert to similarity
                        distance = chroma_query_results['distances'][0][i]
                        similarity_score = 1 - distance
                        logger.info(f"Using distance-based fallback similarity score: {similarity_score}")
                    else:
                        similarity_score = 0.5  # Default fallback
                        logger.info(f"Using default fallback similarity score: {similarity_score}")
            
            # Apply similarity threshold
            if similarity_score < similarity_threshold:
                logger.info(f"Skipping chunk {doc_id} due to low similarity score: {similarity_score}")
                continue

            # Apply date filter if provided
            if start_date_str and end_date_str:
                try:
                    doc_date = datetime.fromisoformat(metadata['timestamp']).date()
                    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
                    end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()

                    if not (start_date <= doc_date <= end_date):
                        logger.info(f"Skipping chunk {doc_id} due to date filter mismatch.")
                        continue
                except ValueError:
                    logger.warning(f"Could not parse extracted date(s): start_date={start_date_str}, end_date={end_date_str}")

            # Additional LLM-based relevance check for high-precision filtering
            # try:
            #     # Implement retry logic for Gemini API calls to handle rate limits
            #     max_retries = 5
            #     retry_delay = 1  # seconds
            #     relevance_result = None
            #     for attempt in range(max_retries):
            #         try:
            #             relevance_result = relevance_chain.invoke({"question": question, "content": content[:500]})  # Use first 500 chars for speed
            #             break  # If successful, break the loop
            #         except Exception as e:
            #             logger.warning(f"Attempt {attempt + 1} failed for relevance chain: {e}")
            #             if attempt < max_retries - 1:
            #                 time.sleep(retry_delay)
            #                 retry_delay *= 2  # Exponential backoff
            #             else:
            #                 logger.error(f"Max retries reached for relevance chain. Skipping chunk.")
            #                 continue  # Skip to next chunk if all retries fail

            #     if relevance_result is None:
            #         continue # Skip to next chunk if relevance_result is still None after retries
            #     if relevance_result.get('is_relevant', False) and relevance_result.get('relevance_score', 0) > 0.3:
            #         relevant_chunks.append({
            #             'content': content,
            #             'metadata': metadata,
            #             'similarity_score': similarity_score,
            #             'relevance_score': relevance_result.get('relevance_score', 0)
            #         })
            #         logger.info(f"Chunk {doc_id} passed relevance check with score: {relevance_result.get('relevance_score', 0)}")
            #     else:
            #         logger.info(f"Chunk {doc_id} failed LLM relevance check")
            # except Exception as e:
            #     logger.warning(f"Error in relevance check for chunk {doc_id}: {e}")
            #     # Fallback to similarity-based filtering if LLM check fails
            #     relevant_chunks.append({
            #         'content': content,
            #         'metadata': metadata,
            #         'similarity_score': similarity_score,
            #         'relevance_score': similarity_score
            #     })

        # Sort by combined relevance score
        # relevant_chunks.sort(key=lambda x: (x['relevance_score'] + x['similarity_score']) / 2, reverse=True)
        relevant_chunks.sort(key=lambda x: x['similarity_score'], reverse=True)
        # Take top results˚
        relevant_chunks = relevant_chunks[:max_results]
        
        logger.info(f"Found {len(relevant_chunks)} relevant chunks after filtering.")

        if not relevant_chunks:
            logger.info("No relevant content found after processing query and filters.")
            return jsonify({"message": "No relevant content found for your query.", "urls": []}), 200

        # Prepare content for Gemini
        relevant_contents = [chunk['content'] for chunk in relevant_chunks]
        relevant_urls = []
        
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
        for url, group in url_groups.items():
            # Sort chunks by relevance for this URL
            group['chunks'].sort(key=lambda x: (x['relevance_score'] + x['similarity_score']) / 2, reverse=True)
            relevant_urls.append({
                'url': url,
                'title': group['title'],
                'favicon_url': group['favicon_url'],
                'relevance_score': (group['chunks'][0]['relevance_score'] + group['chunks'][0]['similarity_score']) / 2
            })

        # Sort URLs by relevance
        relevant_urls.sort(key=lambda x: x['relevance_score'], reverse=True)

        # Use Gemini to generate answer
        model = genai.GenerativeModel('gemini-2.5-flash')
        relevant_contents_str = '\n\n'.join([f"Source: {chunk['metadata']['url']}\nContent: {chunk['content']}" for chunk in relevant_chunks])
        
        prompt = f"""Based on the following relevant content, answer the question: {question}

Content:
{relevant_contents_str}

Provide a comprehensive answer based on the most relevant information. If you reference specific information, mention which source it came from.

Question: {question}"""
        
        logger.info("Sending prompt to Gemini model.")
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
            "similarity_threshold_used": similarity_threshold
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