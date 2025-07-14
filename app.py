# app.py
import gradio as gr
import os
import sys

# Add the project root to the system path to import from src
# Assumes app.py is in the project root (e.g., D:/Project/creditrust-rag/app.py)
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.rag_pipeline import RAGPipeline

# --- 1. Initialize RAG Pipeline (Load models, index, LLM) ---
# This part runs once when the Gradio app starts. It will take time to load the LLM.
print("Initializing RAG Pipeline for Gradio app...")

# VECTOR_STORE_PATH should be relative to app.py
VECTOR_STORE_PATH = 'vector_store/' 

# Initialize RAGPipeline with the models you used in Task 3
# Ensure these match the models you successfully loaded previously
rag_pipeline = RAGPipeline(
    model_name="google/gemma-2b-it", # Make sure this matches your loaded LLM
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2", # Make sure this matches
    vector_store_path=VECTOR_STORE_PATH
)

# Load the necessary components
try:
    print("Loading embedding model...")
    rag_pipeline.load_embedding_model()
    print("Loading FAISS index and metadata...")
    rag_pipeline.load_faiss_index()
    print("Loading LLM (this will take time and use significant RAM)...")
    rag_pipeline.load_llm()
    if rag_pipeline.llm is None:
        raise ValueError("Failed to load LLM. Cannot proceed with RAG app.")
    print("Setting up RAG chain...")
    rag_pipeline.setup_rag_chain() # Ensure this sets up the chain for streaming
    print("RAG Pipeline initialized. App ready!")
except Exception as e:
    print(f"ERROR: Failed to initialize RAG Pipeline. Please ensure all previous tasks were completed successfully and models/index files exist. Error: {e}")
    # In a real app, you might show an error message in the UI or exit gracefully.
    rag_pipeline = None # Indicate that pipeline is not ready

# --- 2. Define Gradio Interface Function (Modified for streaming) ---
def ask_rag_system_stream(question: str):
    """
    Function to query the RAG system and stream the answer, displaying sources.
    """
    if not rag_pipeline:
        yield "The RAG system failed to initialize. Please check the console for errors.", "N/A"
        return
    
    if not question:
        yield "Please enter your question.", "No sources yet."
        return

    try:
        # rag_pipeline.query now yields (answer_chunk, formatted_sources)
        for answer_chunk, sources_text in rag_pipeline.query(question):
            yield answer_chunk, sources_text
    except Exception as e:
        print(f"Error during RAG query: {e}")
        yield f"An error occurred while processing your question: {e}", "Could not retrieve sources due to error."

# --- 3. Define Clear Function ---
def clear_interface():
    """
    Clears the input and output fields in the Gradio interface.
    """
    return "", "", "" # Clear question, answer, and sources

# --- 4. Build Gradio Interface ---
# Using gr.Blocks for more layout control
with gr.Blocks(title="CrediTrust RAG Assistant") as demo:
    gr.Markdown("# CrediTrust Financial Analyst Assistant")
    gr.Markdown(
        "Ask questions related to credit scores, financial advice, or customer complaints. "
        "The AI will provide answers based on its knowledge and retrieved documents."
    )

    with gr.Row():
        with gr.Column(scale=1): # Column for input and submit button
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g., How can I improve my credit score?",
                lines=5, # Allows multi-line input
                interactive=True
            )
            submit_button = gr.Button("Ask CrediTrust", variant="primary")
            clear_button = gr.Button("Clear All", variant="secondary")

        with gr.Column(scale=2): # Column for AI answer and sources
            answer_output = gr.Markdown(label="AI-Generated Answer", value="Your answer will appear here...")
            sources_output = gr.Markdown(label="Retrieved Sources", value="Relevant sources will appear here...")

    # Connect components to functions
    submit_button.click(
        fn=ask_rag_system_stream, # Use the streaming function
        inputs=[question_input],
        outputs=[answer_output, sources_output],
        api_name="predict", # Required for streaming with gr.Interface
        queue=True # Required for streaming
    )
    
    # Clear button functionality
    clear_button.click(
        fn=clear_interface,
        inputs=[],
        outputs=[question_input, answer_output, sources_output]
    )

# --- 5. Launch the Gradio App ---
if __name__ == "__main__":
    # Launch with share=True to get a public URL (useful for sharing/testing remotely)
    # Be aware that public links are temporary and expose your local machine.
    print("\n--- Gradio App is launching ---")
    print("The app will open in your browser or provide a local URL.")
    print("Loading the LLM takes significant time, please be patient after launching.")
    demo.launch(inbrowser=True, show_api=False) # set share=True for a public link