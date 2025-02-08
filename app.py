from flask import Flask, request, jsonify, render_template
import logging
import sys
import os
import requests
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub

# Charger les variables d'environnement
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Initialisation de l'application Flask
app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:3000",  # Pour le développement local
            "http://localhost:5173",  # Pour Vite en développement
            "https://ai-agency-dakar.netlify.app"  # Votre domaine Netlify
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# URL FIXE et UNIQUE pour le RAG
FIXED_URL = "https://www.vendasta.com/content-library/ai-automation-agency-website-example/"
os.environ['USER_AGENT'] = 'YourAppName/1.0'
def clean_documents(documents):
    for doc in documents:
        doc.page_content = doc.page_content.replace("vendasta.com", "")
    return documents

def create_rag_chain():
    try:
        logger.info(f"🔄 Chargement des documents depuis {FIXED_URL}")
        loader = WebBaseLoader(FIXED_URL)
        documents = loader.load()[:2]  # Limite à 2 documents
        logger.info(f"Documents chargés : {len(documents)}")
        documents = clean_documents(documents)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        splits = text_splitter.split_documents(documents)
        logger.info(f"Documents divisés en {len(splits)} chunks.")

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(splits, embeddings)
        logger.info(f"Vectorisation réussie avec FAISS.")

        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""...
            """
        )
        # Créez une instance de votre modèle Hugging Face
        model = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.1", api_key=HF_API_KEY)

        retrieval_qa = RetrievalQA.from_chain_type(
            llm=model,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": prompt_template},
            verbose=True
        )
       

        logger.info("✅ Chaîne RAG prête")
        return retrieval_qa

    except Exception as e:
        logger.error(f"🚨 Erreur RAG : {e}")
        return None


# Initialisation de la chaîne RAG
global_rag_chain = create_rag_chain()

def query_huggingface_api(prompt):
    """Interroge l'API Hugging Face avec un modèle léger."""
    try:
        API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 150, "temperature": 0.5}}
        
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        output = response.json()

        logger.info(f"Réponse Hugging Face: {output}")

        return output[0]["generated_text"] if output else "Aucune réponse générée."
    except Exception as e:
        logger.error(f"🚨 Erreur API Hugging Face: {e}")
        return "Erreur lors de la génération de texte."


@app.route("/chat", methods=["POST"])
def chat():
    try:
        if global_rag_chain is None:
            logger.error("Chaîne RAG non initialisée")
            return jsonify({"message": "Le modèle RAG n'a pas pu être initialisé."}), 500

        data = request.json
        user_message = data.get("message", "")
        logger.info(f"💬 Message reçu : {user_message}")

        # Recherche dans la base RAG
        retrieval_response = global_rag_chain.invoke({"query": user_message})
        logger.info(f"💬 Réponse récupération contextuelle: {retrieval_response}")
        context = retrieval_response["result"]

       

        # Génération via Hugging Face API
        full_prompt = f"Context: {context}\nQuestion: {user_message}\nRéponse:"
        model_response = query_huggingface_api(full_prompt)

        return jsonify({"response": model_response})

    except Exception as e:
        logger.error(f"🚨 Erreur Chat : {e}")
        return jsonify({"response": f"Erreur : {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)


