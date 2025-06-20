🤖 AI Chatbot with LangChain + Streamlit

This is a conversational AI chatbot powered by LangChain, using either an LLM-only or Retrieval-Augmented Generation (RAG) approach. The app runs on a Streamlit interface for easy interaction and supports chat memory, and optionally vector store search using FAISS or MongoDB.

🏗️ Architecture Overview

            +-------------------------+
            |      Streamlit UI      |
            +-----------+-------------+
                        |
                        v
            +-------------------------+
            |  LangChain Chain (LLM)  |
            |  - ConversationChain    |
            |  - OR ConversationalRAG |
            +-----------+-------------+
                        |
            +-----------v-----------+
            |     Memory Module     | ← Maintains chat history
            +-----------------------+
                        |
            +-----------v-----------+
            |  (Optional) Retriever | ← FAISS / MongoDB vector DB
            +-----------------------+

            
LLM: Handles natural language generation.

Memory: Stores ongoing chat history using ConversationBufferMemory.

Retriever (Optional): Enhances answers by fetching relevant documents from a vector DB.

UI: Streamlit interface for easy deployment and interaction.

🛠️ Setup Instructions

1. Clone the Repository


git clone https://github.com/yourusername/your-repo.git
cd your-repo

2. Create and Activate a Virtual Environment

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

3. Install Requirements

pip install -r requirements.txt

4. Set Up Environment Variables

Create a .env file in the project root with the following:

OPENAI_API_KEY=your_openai_api_key
MONGODB_URI=your_mongodb_connection_uri (optional)

💡 You can use any LLM provider (e.g., OpenAI, Cohere, Hugging Face) supported by LangChain.

🚀 Running the Chatbot

streamlit run streamlit.py
This will launch the chatbot in your browser.

🧪 Sample Usage

Type a message like:

"What is LangChain?"

And get a smart answer based on the LLM (and optionally your documents).

📂 File Structure

├── main.py               # Chatbot logic (LangChain chain)
├── streamlit.py          # Streamlit UI
├── requirements.txt
├── .env                  # Environment config
├── README.md


🧠 Tech Stack

Python 3.9+

Streamlit – frontend UI

LangChain – core chain logic

FAISS / MongoDB – vector store (optional)

OpenAI / Hugging Face – LLM provider

dotenv – for secret config

📌 To Do

 Add vector store integration (e.g. FAISS or MongoDB Atlas Vector Search)

 Export chat history

 Add source document highlighting

🙌 Acknowledgments

LangChain

Streamlit

OpenAI
