# RAG (Retrieval-Augmented Generation) Application

This project is a full-stack RAG application consisting of a Chrome Extension, a Python FastAPI Backend, and a Next.js Web UI.

## Project Structure

- `chrome-extension/`: Contains the source code for the Chrome Extension.
- `backend/`: Contains the source code for the Python FastAPI backend.
- `web-ui/`: Contains the source code for the Next.js web user interface.

## Tech Stack

### Backend (Python)

- **Framework**: FastAPI
- **Vector Database**: ChromaDB
- **Database**: MongoDB
- **RAG Framework**: LangChain
- **LLM**: Google Gemini API
- **Embeddings**: OpenAI embeddings or sentence-transformers
- **Dependencies**: `langchain`, `langchain-chroma`, `langchain-google-genai`, `pymongo`, `fastapi`

### Frontend/UI

- **Framework**: Next.js 14 with TypeScript
- **Styling**: Tailwind CSS + shadcn/ui
- **State Management**: React Query

### Chrome Extension

- **Manifest**: V3
- **Features**: Content script and background service worker for web page scraping and storage via API.

## Setup and Running

### Environment Variables
Create a `.env` file in the `backend/` directory with the following variables:

```
GOOGLE_API_KEY=your_google_api_key_here
MONGODB_URI=your_mongodb_connection_string
```

### Backend Setup
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

### Chrome Extension Setup
1. Open Chrome and go to `chrome://extensions/`.
2. Enable "Developer mode" in the top right corner.
3. Click "Load unpacked" and select the `chrome-extension/` directory.
4. The extension icon will appear in your browser toolbar.

### Web UI Setup
1. Navigate to the `web-ui/` directory:
   ```bash
   cd web-ui/
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Run the Next.js development server:
   ```bash
   npm run dev
   ```
   The Web UI will be accessible at `http://localhost:3000`.