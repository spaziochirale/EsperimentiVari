# Spiegazione dettagliata del programma

Questo programma è un chatbot basato su Retrieval Augmented Generation (RAG) che consente di caricare un file PDF, indicizzarne il contenuto e rispondere a domande relative al testo del PDF. Utilizza la libreria **LangChain**, che facilita l'integrazione di modelli di linguaggio di grandi dimensioni (LLM) con diverse fonti di dati e flussi di lavoro.

Di seguito, fornirò una spiegazione dettagliata di ogni parte del programma, inclusi approfondimenti sulle istruzioni di LangChain utilizzate.

---

## Installazione delle dipendenze

Prima di eseguire il programma, è necessario installare le seguenti librerie:

```bash
pip install langchain langchain_community langchain_chroma pypdf langchain-openai
```

- **langchain**: libreria principale per creare catene di LLM.
- **langchain_community**: estensioni della comunità per LangChain.
- **langchain_chroma**: integrazione con il database vettoriale Chroma.
- **pypdf**: per leggere ed estrarre testo dai file PDF.
- **langchain-openai**: per interagire con i modelli OpenAI tramite LangChain.

---

## Importazione dei moduli

```python
import os
import sys
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

- **os** e **sys**: moduli standard di Python per interagire con il sistema operativo.
- **ChatOpenAI**: classe per interagire con i modelli di chat di OpenAI.
- **OpenAIEmbeddings**: per generare embedding dei testi utilizzando OpenAI.
- **hub**: modulo per gestire prompt predefiniti o personalizzati.
- **Chroma**: database vettoriale per memorizzare e recuperare embedding.
- **PyPDFLoader**: per caricare e leggere file PDF.
- **StrOutputParser**: per elaborare l'output dell'LLM in una stringa.
- **RunnablePassthrough**: per passare l'input attraverso una pipeline senza modificarlo.
- **RecursiveCharacterTextSplitter**: per dividere il testo in segmenti più piccoli.

---

## Verifica della chiave API di OpenAI

```python
if "OPENAI_API_KEY" not in os.environ:
    print("Errore: La variabile d'ambiente OPENAI_API_KEY non è impostata.")
    sys.exit(1)
```

- Il programma verifica se la chiave API di OpenAI è impostata come variabile d'ambiente. Se non lo è, termina l'esecuzione con un messaggio di errore.

---

## Creazione dell'oggetto LLM

```python
llm = ChatOpenAI(model="gpt-4o-mini")
```

- **ChatOpenAI**: classe di LangChain per interagire con i modelli di chat di OpenAI.
- **model="gpt-4o-mini"**: specifica il modello di linguaggio da utilizzare. È un modello simile a GPT-4, ma in versione ridotta.

---

## Funzione per caricare il PDF

```python
def load_pdf(file_path):
    if not os.path.exists(file_path):
        print(f"Errore: Il file {file_path} non esiste.")
        sys.exit(1)
    loader = PyPDFLoader(file_path)
    return loader.load()
```

- **load_pdf**: funzione che carica il file PDF dal percorso specificato.
    - **os.path.exists(file_path)**: verifica se il file esiste.
    - **PyPDFLoader(file_path)**: crea un loader per il PDF.
    - **loader.load()**: carica il contenuto del PDF e lo restituisce come lista di documenti.

---

## Caricamento del file PDF

```python
if len(sys.argv) < 2:
    print("Uso: python script.py <percorso_del_file_pdf>")
    sys.exit(1)

pdf_file = sys.argv[1]
docs = load_pdf(pdf_file)
```

- Il programma verifica se è stato fornito un argomento da linea di comando (il percorso del PDF).
- **sys.argv[1]**: prende il primo argomento passato allo script.
- **docs**: contiene i documenti estratti dal PDF.

---

## Divisione del contenuto del PDF in chunk

```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
```

- **RecursiveCharacterTextSplitter**: classe di LangChain che divide il testo in segmenti più piccoli.
    - **chunk_size=1000**: lunghezza massima di ogni segmento in caratteri.
    - **chunk_overlap=200**: numero di caratteri che si sovrappongono tra i segmenti.
- **split_documents(docs)**: applica lo splitter ai documenti estratti dal PDF, restituendo una lista di segmenti.

---

## Creazione del database vettoriale

```python
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
```

- **Chroma.from_documents**: metodo che crea un database vettoriale a partire dai documenti.
    - **documents=splits**: i segmenti di testo ottenuti dallo splitter.
    - **embedding=OpenAIEmbeddings()**: specifica che gli embedding saranno generati utilizzando OpenAI.
- **Chroma**: database vettoriale che memorizza gli embedding dei segmenti di testo per il recupero efficiente.

---

## Creazione del retriever e caricamento del prompt template

```python
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")
```

- **vectorstore.as_retriever()**: crea un retriever dal database vettoriale per cercare segmenti rilevanti.
- **hub.pull("rlm/rag-prompt")**: carica un template di prompt predefinito per le catene RAG.
    - **hub**: modulo di LangChain per gestire prompt predefiniti.
    - **"rlm/rag-prompt"**: nome del prompt template da utilizzare.

---

## Funzione di utilità per formattare i documenti

```python
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
```

- **format_docs**: funzione che unisce il contenuto dei documenti in una singola stringa, separata da doppie nuove righe.
    - **doc.page_content**: attributo che contiene il testo del segmento.

---

## Definizione della catena RAG

```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

Questa parte definisce la pipeline di elaborazione delle domande dell'utente.

### Spiegazione dettagliata:

1. **{"context": retriever | format_docs, "question": RunnablePassthrough()}**:
    - Crea un dizionario con due chiavi:
        - **"context"**: pipeline che prende la domanda dell'utente, la passa al retriever per ottenere i segmenti rilevanti, e poi li formatta.
            - **retriever | format_docs**:
                - **retriever**: utilizza l'embedding della domanda per recuperare segmenti pertinenti.
                - **|**: operatore che passa l'output del retriever a `format_docs`.
                - **format_docs**: formatta i segmenti in una stringa.
        - **"question"**: passa la domanda originale dell'utente senza modificarla.
            - **RunnablePassthrough()**: classe che ritorna l'input così com'è.

2. **| prompt**:
    - Applica il prompt template al dizionario precedente.
    - Il prompt template utilizza le chiavi "context" e "question" per formare il messaggio da inviare all'LLM.

3. **| llm**:
    - Il messaggio generato dal prompt viene passato al modello di linguaggio per ottenere una risposta.

4. **| StrOutputParser()**:
    - L'output dell'LLM viene elaborato per estrarre la risposta come stringa.

### Componenti LangChain utilizzate:

- **retriever**: per recuperare segmenti di testo rilevanti dal database vettoriale.
- **RunnablePassthrough()**: per passare l'input senza modifiche.
- **prompt**: template che struttura il messaggio per l'LLM.
- **llm**: modello di linguaggio che genera la risposta.
- **StrOutputParser()**: per estrarre la risposta dall'output dell'LLM.

---

## Funzione principale per eseguire le query

```python
def run_query(question):
    response = rag_chain.invoke(question)
    print(f"Domanda: {question}")
    print(f"Risposta: {response}\n")
```

- **run_query**: funzione che prende la domanda dell'utente, la passa attraverso la catena RAG e stampa la risposta.
    - **rag_chain.invoke(question)**: esegue la catena RAG con la domanda fornita.

---

## Loop di interazione con l'utente

```python
if __name__ == "__main__":
    print(f"PDF caricato: {pdf_file}")
    print("Inserisci le tue domande. Digita 'exit' per uscire.")
    
    while True:
        question = input("Domanda: ")
        if question.lower() == 'exit':
            break
        run_query(question)
    
    print("Grazie per aver usato il chatbot RAG!")
```

- Il programma entra in un loop dove attende le domande dell'utente.
- **input("Domanda: ")**: legge la domanda inserita dall'utente.
- Se l'utente digita **'exit'**, il loop termina.
- Altrimenti, la domanda viene passata a **run_query(question)** per ottenere e stampare la risposta.

---

## Come funziona il programma nel suo insieme

1. **Caricamento del PDF**: il programma carica il file PDF specificato e ne estrae il contenuto testuale.

2. **Pre-elaborazione del testo**:
    - Il testo estratto viene suddiviso in segmenti (chunk) utilizzando **RecursiveCharacterTextSplitter**.
    - Questo aiuta a gestire testi lunghi e a migliorare l'efficienza del recupero.

3. **Creazione degli embedding**:
    - Per ogni segmento, viene calcolato un embedding utilizzando **OpenAIEmbeddings()**.
    - Gli embedding sono rappresentazioni numeriche che catturano il significato semantico del testo.

4. **Creazione del database vettoriale**:
    - Gli embedding vengono memorizzati in un database vettoriale **Chroma**, che consente il recupero rapido dei segmenti pertinenti.

5. **Interazione con l'utente**:
    - L'utente inserisce una domanda relativa al contenuto del PDF.

6. **Recupero dei segmenti rilevanti**:
    - Il **retriever** utilizza l'embedding della domanda per trovare i segmenti di testo più pertinenti nel database vettoriale.

7. **Formattazione del contesto**:
    - I segmenti recuperati vengono formattati in una stringa utilizzando **format_docs**.

8. **Preparazione del prompt**:
    - Viene creato un messaggio utilizzando il **prompt template**, che include sia il contesto (i segmenti rilevanti) sia la domanda dell'utente.

9. **Generazione della risposta**:
    - Il messaggio viene passato all'LLM (**llm**), che genera una risposta basata sia sul contesto che sulla domanda.

10. **Output della risposta**:
    - La risposta dell'LLM viene elaborata da **StrOutputParser()** e stampata all'utente.

---

## Approfondimenti sulle istruzioni LangChain utilizzate

- **LangChain**: una libreria che facilita la creazione di catene modulari per LLM, permettendo di combinare diverse componenti come modelli di linguaggio, prompt, database vettoriali e altro.

- **ChatOpenAI**: interfaccia per interagire con i modelli di chat di OpenAI all'interno di LangChain.

- **OpenAIEmbeddings**: genera embedding dei testi utilizzando i modelli di OpenAI. Gli embedding sono fondamentali per rappresentare testi in uno spazio vettoriale e consentire confronti semantici.

- **Chroma**: un database vettoriale che memorizza gli embedding e consente operazioni di ricerca e recupero efficienti.

- **Retriever**: componente che, dato un input (come una domanda), recupera i documenti o segmenti più rilevanti dal database vettoriale.

- **Prompt Template**: una struttura predefinita che definisce come combinare il contesto e la domanda per creare un messaggio da inviare all'LLM.

- **RunnablePassthrough**: un'utility che consente di passare l'input attraverso una pipeline senza modificarlo, utile quando si devono combinare più flussi di dati.

- **StrOutputParser**: parser che elabora l'output dell'LLM per estrarre la risposta in formato stringa.

- **Pipeline con l'operatore '|'**: in LangChain, l'operatore pipe viene utilizzato per concatenare le operazioni, passando l'output di una funzione come input alla successiva.

---

## Conclusione

Questo programma utilizza le capacità di LangChain per creare un chatbot avanzato che può rispondere a domande specifiche basate sul contenuto di un file PDF. Integra diverse componenti come il caricamento e la suddivisione del testo, la creazione di embedding, il recupero di informazioni pertinenti e la generazione di risposte tramite un LLM.

Le istruzioni di LangChain semplificano la gestione di queste componenti, permettendo di costruire pipeline complesse in modo modulare e leggibile.
