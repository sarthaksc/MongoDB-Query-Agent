
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain.chains import LLMChain
from dotenv import load_dotenv
import textwrap
import re
from datetime import datetime
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import WikipediaLoader,PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
import os
from langchain.chains import ConversationChain
from pymongo import MongoClient


class ChatBot:



    def __init__(self):
        load_dotenv()
        repo_id = "mistralai/Mistral-7B-Instruct-v0.3" # Or any other code-capable model
        self.llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            temperature=0.3,
            top_p=0.95,
            top_k=50,
            huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN")
        )

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="question"  # ✅ Tells memory this is the user input
        )

        self.prompt = PromptTemplate(
            input_variables=["question", "context"],
            template=textwrap.dedent("""\
                You are a MongoDB assistant for the `sample_analytics` database with the following collections:

                - accounts: _id, account_id (int), limit (int), products (array)
                - customers: _id, username, name, email, account_id, birthdate, address
                - transactions: _id, account_id, amount, transaction_code, symbol, date

                You must:
                - Assume `db` is a valid MongoDB database connection object.
                - Only use **modern PyMongo methods** like `count_documents()` instead of deprecated `.count()`.
                - Do not use `MongoClient()` or `print()`.
                - Assume `datetime` is already imported.
                - Use `db.collection.count_documents({{}})` directly to count matching documents.
                - Use datetime(YYYY, MM, DD) to construct dates, e.g., datetime(1970, 1, 1)

                Generate only the **PyMongo logic** (e.g., `db.collection.find(...)` or `aggregate(...)`).

                Question: {question}
                Context: {context}
                PyMongo Code:
            """)
        )

        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory,
            verbose=True
        )
        self.mongo_client = MongoClient("mongodb+srv://sarthaksc:jBNJg0nbqW14qvrr@cluster0.v041joh.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
        self.db = self.mongo_client["sample_analytics"]

    def extract_code(self, llm_response):
        code_blocks = re.findall(r"```python(.*?)```", llm_response, re.DOTALL)
        return code_blocks[0].strip() if code_blocks else None

    def execute_query(self, code: str):
        safe_globals = {"db": self.db, "datetime": datetime}
        safe_locals = {}

        try:
            exec(code, safe_globals, safe_locals)


            if safe_locals:

                for key in reversed(list(safe_locals.keys())):
                    if not key.startswith("__"):
                        return safe_locals[key]

            return "Query executed but no result variable found."
        except Exception as e:
            return f"Error executing query: {e}"

    def ask(self, question):
        # Step 1: Let LLM generate a PyMongo query
        query_response = self.chain.invoke({"question": question, "context": ""})
        mongo_code = self.extract_code(query_response['text'])

        if not mongo_code:
            return "❌ Could not extract code from LLM response."


        result = self.execute_query(mongo_code)


        final_response = self.chain.invoke({
            "question": f"{question} (MongoDB query result: {result})",
            "context": f"The query returned this result: {result}"
        })

        return {
            "generated_code": mongo_code,
            "query_result": result,
            "final_answer": final_response['text']
        }
