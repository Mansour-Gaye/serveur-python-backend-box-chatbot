from flask import Flask, request, jsonify, render_template
import logging
import sys
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI  # Remplace Ollama
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from flask_cors import CORS
import os
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv()

# Configuration du logging avec rotation des fichiers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app_log.txt', maxBytes=10485760, backupCount=5)
    ]
)
logger = logging.getLogger(__name__)

# Initialisation de l'application Flask
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": os.getenv("NETLIFY_URL", "*")}})

def create_rag_chain():
    try:
        # Configuration des chemins et clés API
        PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY", "./chroma_db")
        HF_API_KEY = os.getenv("HUGGING_FACE_API_KEY")
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        FIXED_URL = os.getenv("FIXED_URL")

        if not all([HF_API_KEY, OPENAI_API_KEY, FIXED_URL]):
            raise ValueError("Missing required environment variables")

        # Création du dossier de persistance si nécessaire
        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

        # Vérifier si un vectorstore existe déjà
        if os.path.exists(PERSIST_DIRECTORY) and len(os.listdir(PERSIST_DIRECTORY)) > 0:
            logger.info("Loading existing vectorstore...")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                api_key=HF_API_KEY
            )
            vectorstore = Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=embeddings
            )
        else:
            logger.info("Creating new vectorstore...")
            # Charger et traiter les documents
            loader = WebBaseLoader(FIXED_URL)
            documents = loader.load()
            
            # Diviser les documents en chunks plus petits
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                length_function=len
            )
            splits = text_splitter.split_documents(documents)

            # Créer les embeddings et le vectorstore
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                api_key=HF_API_KEY
            )
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=PERSIST_DIRECTORY
            )

        # Initialiser le modèle LLM (OpenAI au lieu d'Ollama)
        llm = OpenAI(
            api_key=OPENAI_API_KEY,
            temperature=0.5,
            max_tokens=150
        )

        # Template de prompt optimisé
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            En tant qu'assistant virtuel professionnel, répondez à la question
            en vous basant uniquement sur le contexte fourni.
            
            Contexte : {context}
            Question : {question}
            Réponse concise :
            """
        )

        # Créer la chaîne RAG avec gestion de la mémoire
        retrieval_qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_kwargs={"k": 3}  # Limite le nombre de documents récupérés
            ),
            chain_type_kwargs={
                "prompt": prompt_template,
                "verbose": True
            }
        )

        logger.info("RAG chain created successfully")
        return retrieval_qa

    except Exception as e:
        logger.error(f"Error creating RAG chain: {e}")
        return None

# Route de santé pour Render
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

# Le reste du code reste identique...
