# Installazione delle dipendenze:
# pip install langchain langchain_community langchain_chroma pypdf langchain-openai

import os
import sys
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Verifica che la chiave API di OpenAI sia impostata
if "OPENAI_API_KEY" not in os.environ:
    print("Errore: La variabile d'ambiente OPENAI_API_KEY non è impostata.")
    sys.exit(1)

# Creazione dell'oggetto llm
llm = ChatOpenAI(model="gpt-4o-mini")

# Funzione per caricare il PDF
def load_pdf(file_path):
    if not os.path.exists(file_path):
        print(f"Errore: Il file {file_path} non esiste.")
        sys.exit(1)
    loader = PyPDFLoader(file_path)
    return loader.load()

# Caricamento del file PDF
if len(sys.argv) < 2:
    print("Uso: python script.py <percorso_del_file_pdf>")
    sys.exit(1)

pdf_file = sys.argv[1]
docs = load_pdf(pdf_file)

# Divisione del contenuto del PDF in chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Creazione del database vettoriale
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Creazione del retriever e caricamento del prompt template
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

# Funzione di utilità per formattare i documenti
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Definizione della chain RAG
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Funzione principale per eseguire le query
def run_query(question):
    response = rag_chain.invoke(question)
    print(f"Domanda: {question}")
    print(f"Risposta: {response}\n")

# Esempio di utilizzo
if __name__ == "__main__":
    print(f"PDF caricato: {pdf_file}")
    print("Inserisci le tue domande. Digita 'exit' per uscire.")
    
    while True:
        question = input("Domanda: ")
        if question.lower() == 'exit':
            break
        run_query(question)

    print("Grazie per aver usato il chatbot RAG!")
