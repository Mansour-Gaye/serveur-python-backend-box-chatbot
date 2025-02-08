from flask import Flask, request, jsonify
import logging
import sys
import os
from flask_cors import CORS
from dotenv import load_dotenv
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import torch

# Charger les variables d'environnement
load_dotenv()

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Initialisation de Flask
app = Flask(__name__)
CORS(app)

# Récupération des variables d'environnement
HF_API_KEY = os.getenv("HF_API_KEY")  # Clé API Hugging Face
FIXED_URL = os.getenv("FIXED_URL", "https://ai-agency-dakar.netlify.app")
PERSIST_DIRECTORY = "/tmp/chroma_db"  # Render autorise /tmp

# Sélection d'un modèle léger pour Hugging Face
MODEL_NAME = "facebook/opt-125m"

# Chargement du modèle Hugging Face avec la clé API
def init_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HF_API_KEY)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, use_auth_token=HF_API_KEY)
        return pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)  # CPU
    except Exception as e:
        logger.error(f"Erreur d'initialisation du modèle: {e}")
        return None

pipe = init_model()

# Initialisation du RAG
def create_rag_chain():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

        # Vérifier si ChromaDB existe déjà
        if os.path.exists(PERSIST_DIRECTORY):
            return Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)

        # Charger et vectoriser les documents
        loader = WebBaseLoader(FIXED_URL)
        documents = loader.load()[:2]  # Charger moins de documents pour Render

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        splits = text_splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=PERSIST_DIRECTORY)

        return vectorstore

    except Exception as e:
        logger.error(f"Erreur RAG : {e}")
        return None

vectorstore = create_rag_chain()

@app.route("/health")
def health():
    return jsonify({"status": "healthy"}), 200

@app.route("/chat", methods=["POST"])
def chat():
    try:
        if not vectorstore or not pipe:
            return jsonify({"error": "Modèle ou RAG non initialisé"}), 500

        data = request.json
        user_message = data.get("message", "")

        # Récupération du contexte
        docs = vectorstore.similarity_search(user_message, k=1)
        context = docs[0].page_content if docs else ""

        # Génération de réponse
        prompt = f"Context: {context}\nQuestion: {user_message}\nRéponse:"
        response = pipe(prompt, max_new_tokens=50, temperature=0.7)
        answer = response[0]["generated_text"]

        return jsonify({"response": answer})

    except Exception as e:
        logger.error(f"Erreur: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

