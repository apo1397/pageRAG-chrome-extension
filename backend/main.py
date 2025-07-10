from dotenv import load_dotenv
load_dotenv()
import os
import json
from flask import Flask, request, jsonify, Response
from typing import Optional
from flask_cors import CORS
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime, timedelta
import google.generativeai as genai

from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI


from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Initialize LLM for date parsing
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
print(f"GOOGLE_API_KEY from env: {GOOGLE_API_KEY}")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

date_parser_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0, google_api_key=GOOGLE_API_KEY)

def parse_date_query(query: str) -> tuple[Optional[datetime], Optional[datetime]]:
    """
    Parses a natural language date query using a Langchain LLM to extract start and end dates.

    Args:
        query (str): The natural language query containing date information.

    Returns:
        tuple[Optional[datetime], Optional[datetime]]: A tuple containing the parsed start_date
        and end_date as datetime objects. Returns (None, None) if parsing fails or no date
        information is found.

    Raises:
        json.JSONDecodeError: If the LLM response cannot be parsed as valid JSON.
        ValueError: If the date strings from the LLM response are not in the expected format.
    """
    prompt_template = PromptTemplate(
        input_variables=["query", "today_date"],
        template=(
            "You are a helpful assistant that extracts date ranges from user queries.\n"
            "If the query contains a date or date range (e.g., 'yesterday', 'last week', 'last month', 'last year', '2023-01-01'), "
            "extract the start and end dates in YYYY-MM-DD format.\n"
            "If only a single date is mentioned, the start and end dates should be the same.\n"
            "If no date information is present, return {{'start_date': null, 'end_date': null}}.\n"
            "Return the dates as a JSON object with 'start_date' and 'end_date' keys.\n"
            "Example: {{'start_date': '2023-01-01', 'end_date': '2023-01-07'}}\n"
            "Example: {{'start_date': '2023-01-01', 'end_date': '2023-01-01'}}\n"
            "Example: {{'start_date': null, 'end_date': null}}\n"
            "Today's date is {today_date}.\n"
            "Query: {query}"
        )
    )
    
    chain = prompt_template | date_parser_llm
    response = chain.invoke({"query": query, "today_date": datetime.now().strftime('%Y-%m-%d')})
    print(f"Raw LLM response: {response}")
    try:
        # Attempt to parse the JSON response
        response_content = response.content.strip()
        if response_content.startswith('```json') and response_content.endswith('```'):
            response_content = response_content[len('```json'):-len('```')].strip()
        # Attempt to fix common LLM JSON formatting issues (e.g., single quotes)
        response_content = response_content.replace("'", '"')
        date_info = json.loads(response_content)
        start_date_str = date_info.get('start_date')
        end_date_str = date_info.get('end_date')

        start_date = datetime.strptime(start_date_str, '%Y-%m-%d') if start_date_str else None
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d') if end_date_str else None
        logger.info(f"Parsed date range: {start_date} to {end_date}")
        return start_date, end_date
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse date LLM response as JSON: {response} - {e}")
        return None, None
    except ValueError as e:
        logger.error(f"Failed to parse date string from LLM response: {response} - {e}")
        return None, None




# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger.addHandler(handler)

app = Flask(__name__)
CORS(app)

# Initialize Google Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

# Initialize Chroma DB
# Ensure the directory for Chroma DB exists
CHROMA_DB_PATH = "./chroma_db"
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

# Initialize MongoDB
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "rag_extension_db")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "pages")

try:
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]
    pages_collection = db[MONGO_COLLECTION_NAME]
    logger.info("Successfully connected to MongoDB")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    pages_collection = None # Set to None if connection fails

# Function to get vector store
def get_vector_store(text_chunks: list) -> Chroma:
    """
    Initializes and persists a Chroma vector store from a list of text chunks.

    Args:
        text_chunks (list): A list of Document objects to be added to the vector store.

    Returns:
        Chroma: The initialized and persisted Chroma vector store.
    """
    vector_store = Chroma.from_documents(text_chunks, embeddings, persist_directory=CHROMA_DB_PATH)
    vector_store.persist()
    return vector_store

@app.route("/test_query_single_url", methods=["POST"])
def test_query_single_url():
    """
    Tests querying ChromaDB for a single URL and question.

    Args:
        url (str): The URL to filter by.
        question (str): The question to ask.

    Returns:
        str: The answer generated by the conversational chain.
    """
    data = request.json
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)

    search_kwargs = {"filter": {"source": data.get('url')}}
    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
    qa_chain = get_conversational_chain(retriever)

    response = qa_chain.invoke({"input": data.get('question')})
    answer = response["answer"]
    source_documents = response.get("context", [])

    logger.info(f"Test Query for URL: {data.get('url')}, Question: {data.get('question')}")
    logger.info(f"Generated answer: {answer}")
    logger.info(f"Retrieved docs: {str(source_documents)}")
    return jsonify({"answer": answer}),200

# Function to get conversational chain
from langchain.schema.retriever import BaseRetriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

def get_conversational_chain(retriever: BaseRetriever):
    """
    Initializes and returns a conversational retrieval QA chain.

    Args:
        retriever (BaseRetriever): The retriever to be used for the conversational chain.

    Returns:
        RetrievalQA: The initialized conversational retrieval QA chain.
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, google_api_key=GOOGLE_API_KEY)
    system_prompt = (
        "You are a helpful AI assistant. Use the following pieces of context to answer the question at the end. "
        "If you don't know the answer, just say that you don't know, don't try to make up an answer." 
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(model, prompt)
    qa_chain = create_retrieval_chain(retriever, question_answer_chain)

    return qa_chain

# Request models
class PageContent:
    """
    Represents the content of a web page.
    """
    def __init__(self, url: str, content: str):
        """
        Initializes a PageContent object.

        Args:
            url (str): The URL of the web page.
            content (str): The content of the web page.
        """
        self.url = url
        self.content = content

class Query:
    """
    Represents a user query.
    """
    def __init__(self, question: str):
        """
        Initializes a Query object.

        Args:
            question (str): The user's question.
        """
        self.question = question

@app.route("/save_page", methods=["POST"])
def save_page() -> tuple[Response, int]:
    """
    Saves web page content to a vector database and its metadata to MongoDB.

    The function expects a JSON payload with 'url' and 'content' fields, and an optional 'title' field.
    It processes the content, splits it into chunks, and stores it in a Chroma vector store.
    Page metadata (title, URL, and current date) is also stored in MongoDB.

    Returns:
        tuple[Response, int]: A Flask Response object and an HTTP status code.
            - 200 OK: If the page content is saved and processed successfully.
            - 400 Bad Request: If the request is not JSON or is missing required fields.
            - 500 Internal Server Error: If an error occurs during processing (e.g., database error).
    """
    logger.info("Received request to save page content")
    
    if not request.is_json:
        logger.error("Invalid request: Content-Type must be application/json")
        return jsonify({"error": "Content-Type must be application/json"}), 400
    
    try:
        data = request.get_json()
        if not data or 'url' not in data or 'content' not in data:
            logger.error("Invalid request: Missing required fields")
            return jsonify({"error": "Missing required fields: url and content"}), 400
            
        page_content = PageContent(url=data['url'], content=data['content'])
        page_title = data.get('title', page_content.url) # Get title from payload, or use URL as fallback
        logger.info(f"Processing content from URL: {page_content.url}")

        # Determine the date to use
        date_str = data.get('date')
        page_date = datetime.now()
        if date_str:
            try:
                page_date = datetime.strptime(date_str, '%d/%m/%Y')
            except ValueError:
                logger.warning(f"Invalid date format provided: {date_str}. Using current time instead.")
        
        if pages_collection:
            try:
                # Store metadata in MongoDB
                page_metadata = {
                    "title": page_title,
                    "url": page_content.url,
                    "date": page_date,
                    "favicon_url": data.get('favicon_url')
                }
                pages_collection.insert_one(page_metadata)
                logger.info(f"Page metadata saved to MongoDB: {page_metadata}")
            except Exception as mongo_e:
                logger.error(f"Failed to save page metadata to MongoDB: {mongo_e}")

        from langchain.schema import Document
        document = Document(page_content=page_content.content, metadata={"source": page_content.url, "title": page_title})

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        text_chunks = text_splitter.split_documents([document])
        logger.info(f"Split content into {len(text_chunks)} chunks")

        vector_store = get_vector_store(text_chunks)
        logger.info("Successfully saved content to vector database")
        
        return jsonify({"message": "Page content saved and processed successfully!"}), 200
        
    except Exception as e:
        logger.error(f"Error saving page content: {str(e)}", exc_info=True)
        return jsonify({"error": "An error occurred while processing your request"}), 500

@app.route("/query_pages", methods=["POST"])
def query_pages() -> tuple[Response, int]:
    """
    Queries the vector database for answers based on a natural language question, optionally filtering by date.

    The function expects a JSON payload with a 'question' field. It uses an LLM to parse
    any date information within the question to filter results from MongoDB before querying
    the Chroma vector store.

    Returns:
        tuple[Response, int]: A Flask Response object and an HTTP status code.
            - 200 OK: If the query is processed successfully, with the answer or a message
                      indicating no documents were found for the specified date.
            - 400 Bad Request: If the request is not JSON or is missing the 'question' field.
            - 500 Internal Server Error: If an error occurs during processing (e.g., MongoDB error).
    """
    logger.info("Received query request")
    
    if not request.is_json:
        logger.error("Invalid request: Content-Type must be application/json")
        return jsonify({"error": "Content-Type must be application/json"}), 400
    
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            logger.error("Invalid request: Missing question field")
            return jsonify({"error": "Missing required field: question"}), 400
            
        question = data['question']

        logger.info(f"Processing query: {question}")

        filtered_urls = None
        if pages_collection:
            start_date, end_date = parse_date_query(question)
            
            if start_date and end_date:
                # Adjust end_date to include the entire day
                end_date = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)
                
                mongo_query = {
                    "date": {
                        "$gte": start_date,
                        "$lte": end_date
                    }
                }
                try:
                    matching_pages = pages_collection.find(mongo_query, {"url": 1, "_id": 0})
                    filtered_urls = [page["url"] for page in matching_pages]
                    logger.info(f"Filtered URLs before check: {filtered_urls}")
                    logger.info(f"Found {len(filtered_urls)} URLs matching date range {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")
                    
                    if not filtered_urls:
                        logger.info("No documents found for the specified date filter.")
                        return jsonify({"answer": "No documents found for the specified date."}), 200
                except Exception as e:
                    logger.error(f"Error filtering documents by date: {e}", exc_info=True)
                    return jsonify({"error": "An error occurred while filtering documents by date."}), 500
            else:
                logger.info(f"No date information found in query: '{question}'. Proceeding without date filter.")



        vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
        
        if not vector_store:
            logger.error("No documents found in the knowledge base")
            return jsonify({"error": "No documents found in the knowledge base."}), 404

        # If filtered_urls exist, we need to filter the retrieval results
        # If filtered_urls exist, we need to filter the retrieval results
        # A better approach would be to use Chroma's metadata filtering directly if supported by Langchain's retriever.
        # For now, we'll pass the filter to the retriever.
        search_kwargs = {}
        if filtered_urls is not None:
            search_kwargs["filter"] = {"source": {"$in": filtered_urls}}
            logger.info(f"Applying filter to retriever: {search_kwargs}")
            retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
            logger.info(f"Retriever with filter: {retriever}")
        else:
            logger.info("No filter applied to retriever.")
            retriever = vector_store.as_retriever()

        qa_chain = get_conversational_chain(retriever) # Pass the retriever to the conversational chain
        logger.info(f"Initialized conversational retrieval QA chain: {qa_chain}")
        response = qa_chain.invoke({"input": question})
        answer = response["answer"]
        logger.info(f"Generated answer: {answer}")
        source_documents = response.get("context", [])
        logger.info(f"Retrieved docs: {str(source_documents)[:50]}")
        
        unique_urls = {}
        for doc in source_documents:
            url = doc.metadata.get('source')
            favicon_url = doc.metadata.get('favicon_url')
            if url:
                if url not in unique_urls:
                    unique_urls[url] = {'url': url, 'favicon_url': favicon_url}
                elif favicon_url and not unique_urls[url].get('favicon_url'):
                    unique_urls[url]['favicon_url'] = favicon_url
        
        unique_urls_list = list(unique_urls.values())
        logger.info(f"Source URLs with favicons: {unique_urls_list}")

        return jsonify({"answer": answer, "source_urls": unique_urls_list}), 200
         
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return jsonify({"error": "An error occurred while processing your query"}), 500

@app.route("/get_saved_pages", methods=["GET"])
def get_saved_pages() -> tuple[Response, int]:
    """
    Retrieves a list of all saved pages from MongoDB.

    Returns:
        tuple[Response, int]: A Flask Response object and an HTTP status code.
            - 200 OK: If the pages are retrieved successfully.
            - 500 Internal Server Error: If an error occurs during processing.
    """
    logger.info("Received request for saved pages")
    try:
        saved_pages = list(pages_collection.find({}, {"_id": 0}).sort("date", -1))
        logger.info(f"Retrieved {len(saved_pages)} saved pages.")
        return jsonify(saved_pages), 200
    except Exception as e:
        logger.error(f"Error retrieving saved pages: {str(e)}", exc_info=True)
        return jsonify({"error": "An error occurred while retrieving saved pages"}), 500


@app.route('/delete_page', methods=['POST'])
def delete_page():
    data = request.json
    url_to_delete = data.get('url')
    if not url_to_delete:
        return jsonify({'error': 'URL is required'}), 400

    if pages_collection is None:
        logger.error("MongoDB connection not established.")
        return jsonify({"error": "Database not available."}), 500

    try:
        # Delete from MongoDB
        result = pages_collection.delete_one({'url': url_to_delete})
        if result.deleted_count > 0:
            logger.info(f"Deleted page from MongoDB: {url_to_delete}")
            message = 'Page deleted successfully'
            status_code = 200
        else:
            logger.warning(f"Page not found for deletion: {url_to_delete}")
            message = 'Page not found'
            status_code = 404

        # Note: Deleting from ChromaDB is more complex as it's not designed for individual document deletion by URL.
        # A full re-embedding of relevant documents might be needed for complete removal from the vector store.
        # For this example, we'll only remove from MongoDB.

        return jsonify({'message': message}), status_code
    except Exception as e:
        logger.error(f"Error deleting page: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500


@app.route("/list_pages", methods=["GET"])
def list_pages():
    """
    List all saved page metadata from MongoDB.
    """
    logger.info("Received request to list pages")
    if pages_collection is None:
        logger.error("MongoDB connection not established.")
        return jsonify({"error": "Database not available."}), 500
    try:
        pages = []
        for page in pages_collection.find({}, {"_id": 0}): # Exclude _id from results
            pages.append(page)
        logger.info(f"Retrieved {len(pages)} pages from MongoDB.")
        return jsonify({"pages": pages}), 200
    except Exception as e:
        logger.error(f"Error listing pages from MongoDB: {str(e)}", exc_info=True)
        return jsonify({"error": "An error occurred while retrieving pages."}), 500

if __name__ == "__main__":
    """
    Test CURL commands:
    
    Save page content:
    curl -X POST http://localhost:8000/save_page \
    -H "Content-Type: application/json" \
    -d '{"url": "https://example.com", "content": "Sample content to save", "title": "Example Page"}'
    
    Query pages:
    curl -X POST http://localhost:8000/query_pages \
    -H "Content-Type: application/json" \
    -d '{"question": "summarise the articles from last week"}'

    # Example with a specific date
    curl -X POST http://localhost:8000/query_pages \
    -H "Content-Type: application/json" \
    -d '{"question": "What did I save on AI on 2023-10-26?"}'

    # Example with yesterday
    curl -X POST http://localhost:8000/query_pages \
    -H "Content-Type: application/json" \
    -d '{"question": "What was saved yesterday?"}'

    List pages:
    curl -X GET http://localhost:8000/list_pages
    """
    # Example usage of the new test function
    # test_query_single_url("https://newatlas.com/crispr-cas12a-gene-editing-multiple-eth-zurich/61068/?utm_source=tldrnewsletter", "What is this article about?")
    app.run(host='0.0.0.0', port=8000, debug=True)