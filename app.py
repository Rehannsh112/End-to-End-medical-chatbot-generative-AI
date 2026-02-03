from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

# Pinecone VectorStore import (works across versions)
try:
    from langchain_pinecone import PineconeVectorStore
except Exception:
    from langchain_pinecone.vectorstores import PineconeVectorStore

from langchain_openai import ChatOpenAI

# Use LCEL instead of langchain.chains (fixes your 'langchain.chains' missing issue)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY missing in .env")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY missing in .env")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# -------------------------
# Load VectorStore + Retriever
# -------------------------
embeddings = download_hugging_face_embeddings()
index_name = "medicalbot"
namespace = "medical"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
    namespace=namespace
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})


# -------------------------
# LLM + Prompt
# -------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.4,
    max_tokens=500
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

rag_chain = (
    {
        "context": retriever | format_docs,
        "input": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)


# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    if not msg:
        return jsonify({"answer": "Please type a question."})

    try:
        answer = rag_chain.invoke(msg)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"answer": f"Error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
