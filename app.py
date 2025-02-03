# requirements.txt
flask==2.0.1
flask-cors==3.0.10
langchain==0.1.0
langchain_community==0.0.10
python-dotenv==0.19.2
gunicorn==20.1.0
transformers==4.34.0
sentence-transformers==2.2.2

# main.py
from flask import Flask, request, jsonify
import logging
import sys
import os
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline
import torch

# Chargement des variables d'environnement
load_dotenv()

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# Initialisation de l'application Flask
app = Flask(__name__)
CORS(app)

# Configuration depuis les variables d'environnement
HF_API_KEY = os.getenv('HF_API_KEY')
FIXED_URL = os.getenv('FIXED_URL', 'https://ai-agency-dakar.netlify.app')
PERSIST_DIRECTORY = os.getenv('PERSIST_DIRECTORY', '/tmp/chroma_db')

# Initialisation du modèle avec gestion de la mémoire
def init_model():
    try:
        # Utiliser un modèle plus léger pour Render
        return pipeline(
            "text-generation",
            model="distilgpt2",
            token=HF_API_KEY,
            device_map='auto'  # Utilise CPU si pas de GPU
        )
    except Exception as e:
        logger.error(f"Erreur d'initialisation du modèle: {e}")
        return None

pipe = init_model()

def create_rag_chain():
    try:
        # Vérifier si le vectorstore existe déjà
        if os.path.exists(PERSIST_DIRECTORY):
            embeddings = HuggingFaceEmbeddings(
                model_name="paraphrase-MiniLM-L3-v2",
                cache_folder=PERSIST_DIRECTORY
            )
            return Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=embeddings
            )

        # Création d'un nouveau vectorstore
        loader = WebBaseLoader(FIXED_URL)
        documents = loader.load()[:3]  # Limite encore plus stricte pour Render
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,  # Chunks plus petits
            chunk_overlap=20
        )
        splits = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="paraphrase-MiniLM-L3-v2",
            cache_folder=PERSIST_DIRECTORY
        )

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY
        )
        
        return vectorstore

    except Exception as e:
        logger.error(f"Erreur RAG : {e}")
        return None

# Initialisation au démarrage
vectorstore = create_rag_chain()

@app.route('/health')
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/chat', methods=['POST'])
def chat():
    try:
        if not vectorstore or not pipe:
            return jsonify({
                'error': "Services non initialisés"
            }), 500

        data = request.json
        user_message = data.get('message', '')
        
        # Récupération du contexte
        docs = vectorstore.similarity_search(user_message, k=1)
        context = docs[0].page_content if docs else ""

        # Génération de la réponse
        prompt = f"Context: {context}\nQuestion: {user_message}\nRéponse:"
        response = pipe(prompt, max_new_tokens=50, temperature=0.7)
        answer = response[0]['generated_text']

        return jsonify({'response': answer})

    except Exception as e:
        logger.error(f"Erreur: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port)

