from flask import Flask, request, jsonify
import logging
import sys
import os
from flask_cors import CORS
from dotenv import load_dotenv
from transformers import pipeline
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import torch

# Chargement des variables d'environnement
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
HF_API_KEY = os.getenv("HF_API_KEY")
FIXED_URL = os.getenv("FIXED_URL", "https://ai-agency-dakar.netlify.app")

# Modèle d'embed léger
embedder = SentenceTransformer("paraphrase-MiniLM-L3-v2")

# Stockage RAG en mémoire
vectorstore = {}

# Chargement et vectorisation des documents
def create_rag_chain():
    global vectorstore
    try:
        loader = WebBaseLoader(FIXED_URL)
        documents = loader.load()[:2]  # Charge moins de docs

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        splits = text_splitter.split_documents(documents)

        vectorstore = {doc.page_content: embedder.encode(doc.page_content) for doc in splits}
        logger.info("RAG chargé en mémoire ✅")

    except Exception as e:
        logger.error(f"Erreur RAG : {e}")
        vectorstore = {}

# Chargement d'un modèle Hugging Face léger
def init_model():
    try:
        return pipeline(
            "text-generation",
            model="facebook/opt-125m",
            token=HF_API_KEY,
            device=0 if torch.cuda.is_available() else -1  # CPU
        )
    except Exception as e:
        logger.error(f"Erreur d'initialisation du modèle: {e}")
        return None

pipe = init_model()
create_rag_chain()

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

        # Recherche du contexte
        query_embedding = embedder.encode(user_message)
        best_match = max(vectorstore.items(), key=lambda x: torch.cosine_similarity(
            torch.tensor(x[1]), torch.tensor(query_embedding), dim=0
        ))

        context = best_match[0] if best_match else ""

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
