from flask import Flask, request, jsonify, render_template
import logging
import sys
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from transformers import pipeline
from flask_cors import CORS

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app_log.txt')
    ]
)
logger = logging.getLogger(__name__)

# Initialisation de l'application Flask
app = Flask(__name__)
CORS(app)

# Initialisation du modèle Hugging Face
HF_API_KEY = "VOTRE_CLE_API_HUGGING_FACE"
pipe = pipeline("text-generation", model="microsoft/phi-2", token=HF_API_KEY)

def clean_documents(documents):
    for doc in documents:
        doc.page_content = doc.page_content.replace("vendasta.com", "")
    return documents

# URL FIXE pour le RAG
FIXED_URL = "https://ai-agency-dakar.netlify.app"

def create_rag_chain():
    try:
        logger.info(f"Chargement des documents depuis {FIXED_URL}")
        
        # 1. Charger et nettoyer les documents
        loader = WebBaseLoader(FIXED_URL)
        documents = loader.load()
        documents = clean_documents(documents)

        # 2. Diviser les documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(documents)

        # 3. Créer les embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # 4. Créer le vectorstore
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

        # 5. Personnaliser le prompt
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            Vous êtes un assistant virtuel représentant une entreprise.
            Répondez de manière concise et professionnelle à la question posée en vous basant uniquement sur le contexte fourni.

            Contexte : {context}
            Question : {question}
            Réponse :
            """
        )

        # 6. Créer la chaîne RAG
        retrieval_qa = RetrievalQA.from_chain_type(
            llm=None,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": prompt_template},
            verbose=True
        )

        logger.info("Chaîne RAG créée avec succès")
        return retrieval_qa

    except Exception as e:
        logger.error(f"Erreur lors de la création de la chaîne RAG : {e}")
        return None

# Créer la chaîne RAG au démarrage
global_rag_chain = create_rag_chain()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        if global_rag_chain is None:
            logger.error("Chaîne RAG non initialisée")
            return jsonify({
                'message': "Le modèle RAG n'a pas pu être initialisé."
            }), 500

        data = request.json
        user_message = data.get('message', '')
        logger.info(f"Message reçu : {user_message}")

        # Utiliser la chaîne RAG pour le contexte
        rag_response = global_rag_chain.invoke({
            "query": user_message
        })
        context = rag_response['result']

        # Générer une réponse avec phi-2
        hf_response = pipe(f"{context}\nQuestion: {user_message}\nRéponse:", max_new_tokens=150, temperature=0.5)
        answer = hf_response[0]['generated_text']

        logger.info(f"Réponse générée : {answer}")
        return jsonify({'response': answer})

    except Exception as e:
        logger.error(f"Erreur lors du traitement du chat : {e}")
        return jsonify({'response': f"Erreur : {str(e)}"}), 500

if __name__ == '__main__':
    logger.info("Démarrage du serveur Flask...")
    app.run(debug=True, host='0.0.0.0', port=8080)
