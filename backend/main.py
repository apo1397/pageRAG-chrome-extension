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
    favicon_url = data.get('favicon_url')

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

        # Generate embeddings and store in ChromaDB
        # Use GoogleGenerativeAIEmbeddings for consistency with Langchain
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        embedding = embedding_model.embed_query(content)
        logger.info(f"Generated embedding for URL hash: {url_hash}")

        collection.add(
            documents=[content],
            embeddings=[embedding],
            metadatas=[{
                "url": url,
                "timestamp": page_metadata["timestamp"].isoformat() if page_metadata.get("timestamp") else None,
                "title": title if title is not None else "",
                "favicon_url": favicon_url if favicon_url is not None else ""
            }],
            ids=[url_hash] # Using URL hash as a unique ID
        )
        logging.info(f"Successfully added document to ChromaDB for URL: {url}")
    except Exception as e:
        logging.error(f"Error adding document to ChromaDB for URL {url}: {e}")
        return jsonify({"error": f"Failed to process content: {e}"}), 500

    logger.info(f"Content and embedding stored in ChromaDB for URL hash: {url_hash}")
    return jsonify({"message": "Content processed successfully", "url": url, "url_hash": url_hash}), 200

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
            n_results=5 # Retrieve top 5 relevant documents
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
        prompt = f"Given the following content:\n\n{relevant_contents_str}\n\nAnswer the question: {question}\n\nProvide a concise answer. Also try to find the URL from which the question could've come."
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