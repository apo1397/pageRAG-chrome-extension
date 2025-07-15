# RAG Chrome Extension: Your Personal Knowledge Base

This project is a full-stack Retrieval-Augmented Generation (RAG) application designed to help you build a personal knowledge base from the web pages you browse. It consists of a Chrome Extension for scraping, a Python FastAPI Backend for data processing and RAG, and a Next.js Web UI for interaction.

## Project Structure

- `chrome-extension/`: Contains the source code for the Chrome Extension, responsible for scraping web content and sending it to the backend.
- `backend/`: Houses the Python FastAPI backend, which handles data storage (MongoDB), vector embeddings (ChromaDB), and RAG functionalities using LangChain and Google Gemini.

## Tech Stack

### Backend (Python)

- **Framework**: FastAPI
- **Vector Database**: ChromaDB
- **Database**: MongoDB
- **RAG Framework**: LangChain
- **LLM**: Google Gemini API
- **Embeddings**: Google Generative AI Embeddings (`models/embedding-001`)
- **Dependencies**: `langchain`, `langchain-chroma`, `langchain-google-genai`, `pymongo`, `fastapi`, `uvicorn`, `python-dotenv`, `numpy`, `scikit-learn`

### Chrome Extension

- **Manifest**: V3
- **Features**: Content script and background service worker for web page scraping and storage via API.

## Setup and Running

### 1. Environment Variables

Create a `.env` file in the `backend/` directory with the following variables:

```
GOOGLE_API_KEY=your_google_api_key_here
MONGODB_URI=your_mongodb_connection_string
```

- **`GOOGLE_API_KEY`**: Obtain this from the Google AI Studio or Google Cloud Console.
- **`MONGODB_URI`**: Your MongoDB connection string (e.g., `mongodb://localhost:27017/rag_db` for a local instance, or a connection string from MongoDB Atlas).

### 2. Backend Setup

1. Navigate to the `backend/` directory:
   ```bash
   cd backend/
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the FastAPI application:
   ```bash
   uvicorn main:app --reload
   ```
   The backend will be accessible at `http://127.0.0.1:8000`.

### 3. Chrome Extension Setup

1. Open Chrome and go to `chrome://extensions/`.
2. Enable "Developer mode" in the top right corner.
3. Click "Load unpacked" and select the `chrome-extension/` directory.
4. The extension icon will appear in your browser toolbar.

## Core Logic Explained

### Save Page Flow (`/save_page`)

When you click the "Save Page" button in the Chrome Extension, the following happens:

1.  **Content Extraction**: The `content.js` script in the Chrome Extension scrapes the current page's URL, title, content, and favicon URL.
2.  **Data Transmission**: This content of the page is sent to the backend's `/save_page` (or `/process_content`) endpoint along with the title and favicon URL.
3.  **Backend Processing (`backend/main.py`)**:
    *   **URL Hashing**: A SHA256 hash of the URL is generated to serve as a unique identifier for the page.
    *   **MongoDB Storage**: The page's metadata (URL, title, content, favicon URL, timestamp) is stored or updated in MongoDB. This acts as the primary record for the saved page.
    *   **ChromaDB Integration (Vector Store)**:
        *   **Chunking**: The page's content is split into smaller, manageable chunks using `RecursiveCharacterTextSplitter`. This ensures that each piece of text is small enough for effective embedding and retrieval.
        *   **Content Cleaning**: Chunks are filtered to remove very short or low-quality content (e.g., mostly non-alphanumeric text) to improve the quality of embeddings.
        *   **Keyword Extraction**: Keywords are extracted from each chunk to enrich its metadata, aiding in more targeted retrieval.
        *   **Embedding Generation**: Each chunk is converted into a numerical vector (embedding) using Google Generative AI Embeddings (`models/embedding-001`).
        *   **Storage**: These embeddings, along with their associated metadata (URL, timestamp, title, favicon URL, chunk index, keywords), are stored in ChromaDB. Existing chunks for the same URL are deleted before adding new ones to prevent duplicates.

### Query Flow (`/query_pages`)

When you ask a question in the Chrome Extension's popup, the following steps are executed:

1.  **Question Transmission**: The question is sent from `popup.js` to the backend's `/query_pages` endpoint.
2.  **Backend Processing (`backend/main.py`)**:
    *   **Date Extraction**: The `date_extraction_chain` (powered by Gemini) analyzes the question to extract any date ranges (e.g., "yesterday," "last week") and converts them into absolute `start_date` and `end_date` for filtering.
    *   **Question Embedding**: The user's question is converted into an embedding using the same Google Generative AI Embeddings model.
    *   **ChromaDB Query**: ChromaDB is queried with the question's embedding to retrieve the most semantically similar content chunks. A `n_results` parameter (e.g., 20) is used to fetch a sufficient number of potential matches.
    *   **Date Filtering**: If date ranges were extracted, the retrieved chunks are further filtered to include only those that fall within the specified date period.
    *   **RAG with Gemini**: The filtered, relevant content chunks are then provided to the Google Gemini model as context. Gemini uses this context to generate a concise answer to the user's question.
    *   **Source URL Aggregation**: The URLs of the source pages from which the relevant content was retrieved are collected and returned along with the answer, allowing users to trace back the information.

## Future Improvements

-   **Enhanced Content Cleaning**: Implement more sophisticated content cleaning techniques to remove boilerplate text, ads, and navigation elements more effectively before chunking.
-   **Advanced Chunking Strategies**: Explore different chunking methods (e.g., sentence-based, paragraph-based, or context-aware chunking) to optimize retrieval accuracy.
-   **Hybrid Search**: Combine vector similarity search (ChromaDB) with keyword-based search (e.g., using keywords stored in MongoDB or a dedicated search index) for more robust retrieval.
-   **User Feedback Loop**: Implement a mechanism for users to provide feedback on the quality of answers, which can be used to fine-tune the RAG system.
-   **Multi-modal RAG**: Extend the system to process and retrieve information from images, videos, or other media types found on web pages.
-   **Personalized Summarization**: Allow users to specify the desired length or style of the generated answers.
-   **Improved Date Handling**: More robust parsing of natural language date queries and handling of timezones.
-   **Frontend Enhancements**: Develop the Next.js Web UI to provide a richer user experience for managing saved pages and interacting with the RAG system.
- Handling cases where there's only 1 date, for eg "before the last 24 hrs"
