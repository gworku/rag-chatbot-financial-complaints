# CrediTrust Complaint-Answering RAG Chatbot

## Project Overview
This project develops an internal AI tool for CrediTrust Financial, leveraging Retrieval-Augmented Generation (RAG) to provide synthesized, evidence-backed answers to natural language questions about customer complaints from the CFPB database.

## Project Structure
```
creditrust-rag/
├── data/                  # Stores raw and processed datasets
├── notebooks/             # Jupyter notebooks for experimentation and detailed steps
├── src/                   # Python scripts for reusable functions and pipeline components
├── vector_store/          # Persisted FAISS index and associated metadata
├── reports/               # Evaluation results, summaries, and detailed reports
├── screenshots/           # UI screenshots (for final task)
├── .gitignore             # Specifies intentionally untracked files to ignore
├── README.md              # Project overview and documentation
└── requirements.txt       # List of project dependencies
```
## Setup and Installation
1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd creditrust-rag
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate # On Windows PowerShell
    # For macOS/Linux: source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run
Detailed instructions will be added here for each task as the project progresses.

### Running Task 1 (EDA & Preprocessing)
* **Via Notebook:** Open `notebooks/01_eda_preprocessing.ipynb` and run all cells.
* **Via Script:** Execute the preprocessing script.
    ```bash
    # This command assumes the preprocess.py script is designed to run the preprocessing
    # You might need to adjust arguments based on how your script is set up.
    python src/preprocess.py
    ```

### Running Task 2 (Embedding & Vector Store Indexing)
* **Via Notebook:** Open `notebooks/02_embedding_chunking.ipynb` and run all cells.
* **Via Script:** Execute the vector store creation script.
    ```bash
    # This command assumes the vector_db.py script is designed to run the indexing
    # Ensure filtered_and_cleaned_complaints.csv from Task 1 is available in data/
    python src/vector_db.py
    ```

## Project Progress and Detailed Tasks

### Task 1: Exploratory Data Analysis (EDA) & Data Preprocessing
**Objective:** To understand the structure, content, and quality of the CFPB complaint data and prepare it for the RAG pipeline.

**Implementation Details:**
* **Data Loading:** Loaded the full CFPB complaint dataset from `data/raw_complaints.csv`.
* **Initial EDA:**
    * Analyzed the distribution of complaints across different "Product" categories, identifying dominant complaint types.
    * Calculated and visualized the length (word count) of the "Consumer complaint narrative" column, noting the presence of very short and very long narratives, and managing missing values.
    * Identified the total number of complaints, including those with and without narrative text.
* **Dataset Filtering:**
    * Filtered the dataset to include only records pertaining to specific financial products relevant to the project, such as "Credit card", "Personal loan", "Buy Now, Pay Later", "Savings account", and "Money transfers". (Note: The filtering specifically included a broader set of related categories to ensure comprehensive coverage, like "Credit reporting or other personal consumer reports" due to their prevalence and relevance to core products).
    * Removed any records where the "Consumer complaint narrative" field was empty or contained only boilerplate text, ensuring that only meaningful narratives are processed.
* **Text Cleaning:** Implemented a robust text cleaning function for the "Consumer complaint narrative" column, which included:
    * Lowercasing all text.
    * Removing special characters, numbers, and common boilerplate phrases (e.g., "XXXX" placeholders).
    * Performing lemmatization to reduce words to their base forms (e.g., "running" to "run").
    * Removing common English stop words (e.g., "the", "is", "a") to focus on meaningful terms for embedding.

**Deliverables:**
* Jupyter Notebook: `notebooks/01_eda_preprocessing.ipynb`
* Python Script: `src/preprocess.py` (containing the core cleaning logic)
* Cleaned and Filtered Dataset: `data/filtered_and_cleaned_complaints.csv`

---

### Task 2: Text Chunking, Embedding, and Vector Store Indexing
**Objective:** To convert the cleaned text narratives into a format suitable for efficient semantic search within the RAG pipeline.

**Implementation Details:**
* **Text Chunking Strategy:**
    * Utilized LangChain's `RecursiveCharacterTextSplitter` to break down long complaint narratives into smaller, contextually relevant chunks.
    * Experimented with `chunk_size` (e.g., 500 characters) and `chunk_overlap` (e.g., 100 characters) to ensure chunks maintain coherence and context across boundaries, justifying the final choice in the interim report.
* **Embedding Model Choice:**
    * Selected `sentence-transformers/all-MiniLM-L6-v2` as the embedding model. This model was chosen for its excellent balance of performance (generating high-quality 384-dimensional embeddings) and efficiency (being relatively small and fast, suitable for CPU inference on large datasets).
* **Embedding Generation & Indexing:**
    * Generated vector embeddings for each text chunk using the chosen Sentence Transformer model.
    * Created a FAISS (Facebook AI Similarity Search) index (`IndexFlatIP`) to store these embeddings. FAISS was chosen for its speed and efficiency in similarity search over large vector datasets.
    * Crucially, alongside each vector, associated metadata (including the `Product` category and `Complaint ID`) was stored. This enables tracing a retrieved text chunk back to its original complaint and product, which is essential for providing evidence-backed answers.

**Deliverables:**
* Jupyter Notebook: `notebooks/02_embedding_chunking.ipynb`
* Python Script: `src/vector_db.py` (containing the core chunking, embedding, and indexing logic)
* Persisted Vector Store: `vector_store/faiss_index.bin` (the FAISS index) and `vector_store/metadata.json` (the associated metadata).

---


### Task 3: Building the RAG Core Logic and Evaluation
**Objective:** To build the retrieval and generation pipeline and, most importantly, to evaluate its effectiveness.

**Implementation Details:**
* **Retriever Implementation:**
    * Created a function that takes a user's question (string) as input.
    * Embeds the question using the same model from Task 2.
    * Performs a similarity search against the vector store to retrieve the top-k (k=5) most relevant text chunks.
* **Prompt Engineering:**
    * Designed a robust prompt template to guide the LLM. The template instructs the model to act as a helpful analyst, use only the provided context, and answer the user's question based on that context.
    * *Example Template:*
        ```
        You are an AI assistant specialized in providing information about credit scores and financial literacy.
        Answer the question based only on the following context, which contains information about credit scores, credit reports, and financial advice.
        If the answer cannot be found in the context, politely state that you don't have enough information.

        Context: {context}
        Question: {question}
        Answer:
        ```
* **Generator Implementation:**
    * Combined the prompt, the user question, and the retrieved chunks into a cohesive input.
    * Sent the combined input to an LLM (e.g., using Hugging Face's pipeline via LangChain).
    * Returns the LLM's generated response.
* **Evaluation:**
    * **Quantitative Evaluation:** Integrated advanced evaluation metrics using `Ragas` (for faithfulness, answer relevancy, context recall, context precision) and `DeepEval` (for faithfulness, answer relevancy, context recall). This provides objective scores for pipeline performance.
    * **Qualitative Evaluation:** Prepared a list of 5-10 representative questions to manually assess the system's responses and retrieved sources.

**Deliverables:**
* Jupyter Notebook: `notebooks/03_rag_core_evaluation.ipynb` (for executing the pipeline and running evaluations)
* Python Module: `src/rag_pipeline.py` (containing the core RAG pipeline logic)
* Evaluation Results: Quantitative metric outputs within the notebook, and a qualitative evaluation table (to be created manually in the final report based on notebook output).

---

### Task 4: Creating an Interactive Chat Interface
**Objective:** To build a user-friendly interface that allows non-technical users to interact with your RAG system.

**Implementation Details:**
* **Framework:** Utilized Gradio to build the web interface, chosen for its simplicity and rapid prototyping capabilities.
* **Core Functionality:** The interface includes:
    * A text input box for the user to type their question.
    * A "Ask CrediTrust" button to submit the query.
    * A display area for the AI-generated answer.
* **Enhancing Trust and Usability (Key Requirements):**
    * **Display Sources:** Below the generated answer, the source text chunks that the LLM used are clearly displayed, crucial for user trust and verification.
    * **Streaming:** Implemented response streaming where the answer appears token-by-token, significantly improving the user experience by providing immediate feedback.
    * **Clear Button:** A "Clear All" button is provided to reset the input and output fields, facilitating new conversations.
* **Code Quality:** The code is designed to be clean, modular, and the UI is developed to be intuitive for end-users.

**Deliverables:**
* Python Script: `app.py` (the Gradio application script)
* UI Screenshots: Screenshots or a GIF of your working application (to be included in your final report).

---
Developed for CrediTrust Financial as a Data & AI Engineer.