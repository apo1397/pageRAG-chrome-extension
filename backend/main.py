import os
import hashlib
import logging
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
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
import re
# from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
app.config["MONGO_URI"] = os.getenv("MONGO_URI", "mongodb://localhost:27017/rag_db")
mongo = PyMongo(app)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Google Generative AI
# gen_ai_model = genai.GenerativeModel('gemini-2.0-flash')

# Initialize Langchain with Gemini
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.4)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,  # Slightly smaller for better precision
    chunk_overlap=150,
    length_function=len,
    separators=["\n\n", "\n", ".", "!", "?", ";", ",", " ", ""]  # Better semantic splitting
)

def generate_chunk_id(url_hash: str, chunk_index: int) -> str:
    """
    Generate a unique ID for each chunk
    """
    return f"{url_hash}_chunk_{chunk_index}"

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

date_extraction_parser = JsonOutputParser()
date_extraction_chain = date_extraction_prompt | llm | date_extraction_parser

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="rag_collection")

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

@app.route('/')
def home():
    return "RAG Backend is running!"

def generate_url_hash(url: str) -> str:
    """
    Generates a SHA256 hash for a given URL.
    """
    return hashlib.sha256(url.encode('utf-8')).hexdigest()

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
    favicon_url = data.get('faviconUrl') or data.get('favicon_url')
    timestamp_str = data.get('timestamp')

    # if timestamp_str:
    #     try:
    #         # Parse the ISO 8601 string back to a datetime object
    #         timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    #     except ValueError:
    #         logger.warning(f"Invalid timestamp format received: {timestamp_str}. Using current UTC time.")
    #         timestamp = datetime.utcnow()
    # else:
    #     timestamp = datetime.utcnow()

    logger.info(f"Received request to process content for URL: {url} and favicon_url: {favicon_url} with timestamp: {timestamp}")
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
            "timestamp": datetime.utcnow() # Use the received or generated timestamp
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
        logger.info(f"content before cleaning {content}")
        # Clean and preprocess content
        # cleaned_content = clean_content(content)
        # logger.info(f"Cleaned content {cleaned_content}")
        # Split content into chunks
        chunks = text_splitter.split_text(content)
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
        # Use GoogleGenerativeAIEmbeddings for consistency with Langchain
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        question_embedding = embedding_model.embed_query(question)
        logger.info("Generated embedding for the question.")

        # Query ChromaDB for relevant content
        chroma_query_results = collection.query(
            query_embeddings=[question_embedding],
            n_results=20 # Retrieve top 5 relevant documents
        )
        logger.info(f"ChromaDB query returned {len(chroma_query_results['ids'][0]) if chroma_query_results['ids'] else 0} results.")

        relevant_urls = []
        relevant_contents = []
        for i, doc_id in enumerate(chroma_query_results['ids'][0]):
            metadata = chroma_query_results['metadatas'][0][i]
            content = chroma_query_results['documents'][0][i]

            # Apply date filter if provided by Langchain
            if start_date_str and end_date_str:
                try:
                    doc_date = datetime.fromisoformat(metadata['timestamp']).date()
                    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
                    end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()

                    if not (start_date <= doc_date <= end_date):
                        logger.info(f"Skipping document {doc_id} due to date filter mismatch. Document date: {doc_date}, Filter range: {start_date} to {end_date}")
                        continue
                except ValueError:
                    logger.warning(f"Could not parse extracted date(s): start_date={start_date_str}, end_date={end_date_str}. Proceeding without date filter.")

            relevant_urls.append({"url": metadata['url'], "favicon_url": metadata.get('favicon_url'), "title": metadata.get('title')})
            relevant_contents.append(content)
        
        logger.info(f"Found {len(relevant_contents)} relevant contents after filtering.")

        if not relevant_contents:
            logger.info("No relevant content found after processing query and filters.")
            return jsonify({"message": "No relevant content found for your query.", "urls": []}), 200

        # Use Gemini to summarize/answer based on relevant content
        model = genai.GenerativeModel('gemini-2.0-flash')
        relevant_contents_str = '\n\n'.join(relevant_contents)
        # Format URLs for the prompt
        formatted_urls = [url_obj['url'] for url_obj in relevant_urls]
        logger.info(f"Formatted URLs for prompt: {formatted_urls}")
        prompt = f"Given the following content:\n\n{relevant_contents_str}\n\nAnswer the question: {question}\n\nProvide a concise answer."
        logger.info("Sending prompt to Gemini model.")
        response = model.generate_content(prompt)
        logger.info("Received response from Gemini model.")
        
        # Ensure unique URLs and their favicons
        unique_urls = {}
        unique_titles = {}
        for url_obj in relevant_urls:
            unique_urls[url_obj['url']] = url_obj.get('favicon_url')
            unique_titles[url_obj['url']] = url_obj.get('title')
        
        # Convert back to list of dictionaries for the frontend
        source_urls_for_frontend = []
        for url, favicon in unique_urls.items():
            source_urls_for_frontend.append({"url": url, "favicon_url": favicon, "title": unique_titles.get(url)})
        logger.info({"answer": response.text, "source_urls": source_urls_for_frontend, "question": question})

        return jsonify({"answer": response.text, "source_urls": source_urls_for_frontend}), 200

    except Exception as e:
        logger.error(f"Error querying content for question '{question}': {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_saved_pages', methods=['GET'])
def get_saved_pages():
    try:
        # Fetch all pages from MongoDB
        # We sort by timestamp to get the most recent pages first
        pages_cursor = mongo.db.pages.find({}).sort("timestamp", -1)
        saved_pages = []
        for page in pages_cursor:
            saved_pages.append({
                "url": page["url"],
                "title": page.get("title", "No Title"), # Use .get for optional fields
                "favicon_url": page.get("favicon_url", ""), # Use .get for optional fields
                "date": page["timestamp"].isoformat() # Convert datetime to ISO format string
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

        # Delete from ChromaDB
        collection.delete(ids=[url_hash])
        logger.info(f"Deleted page with URL hash {url_hash} from ChromaDB.")

        return jsonify({"message": "Page deleted successfully"}), 200
    except Exception as e:
        logger.error(f"Error deleting page with URL {url}: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)