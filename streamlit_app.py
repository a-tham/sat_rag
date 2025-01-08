import streamlit as st
from local_rag import LocalRAG  # Import the LocalRAG class we created earlier
import os
import tempfile

# Page config
st.set_page_config(
    page_title="PDF Knowledge Base",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'rag' not in st.session_state:
    st.session_state.rag = None

def initialize_rag():
    """Initialize the RAG system with Cohere API key."""
    try:
        # Get API key from secrets or user input
        if 'COHERE_API_KEY' in st.secrets:
            cohere_api_key = st.secrets.COHERE_API_KEY
        else:
            cohere_api_key = st.session_state.api_key

        # Initialize RAG system
        st.session_state.rag = LocalRAG(cohere_api_key=cohere_api_key)
        return True
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return False

def process_uploaded_file(uploaded_file):
    """Process an uploaded PDF file."""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        # Process the PDF
        with st.spinner('Processing PDF...'):
            success = st.session_state.rag.process_pdf(
                pdf_path=tmp_path,
                source_name=uploaded_file.name
            )

        # Clean up
        os.unlink(tmp_path)

        if success:
            st.success(f"Successfully processed {uploaded_file.name}")
            return True
        else:
            st.error(f"Failed to process {uploaded_file.name}")
            return False

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return False
    
    # Add this print statement in your process_uploaded_file function
    print("Number of chunks:", len(st.session_state.rag._chunk_text(text)))

def main():
    st.title("📚 PDF Knowledge Base")

    # Sidebar for configuration and file upload
    with st.sidebar:
        st.header("Configuration")

        # API Key input if not in secrets
        if 'COHERE_API_KEY' not in st.secrets:
            api_key = st.text_input(
                "Enter your Cohere API key",
                type="password",
                key="api_key"
            )

        # Initialize RAG system
        if st.session_state.rag is None:
            if st.button("Initialize System"):
                if initialize_rag():
                    st.success("System initialized successfully!")

        # File uploader
        if st.session_state.rag is not None:
            st.header("Upload PDF")
            uploaded_file = st.file_uploader(
                "Choose a PDF file",
                type="pdf",
                help="Upload a PDF document to add to your knowledge base"
            )

            if uploaded_file is not None:
                if st.button("Process PDF"):
                    process_uploaded_file(uploaded_file)

    # Main content area
    if st.session_state.rag is None:
        st.info("Please initialize the system using the sidebar.")
        return

    # Get available sources
    sources = st.session_state.rag.list_sources()

    # Source selection
    source_filter = None
    if sources:
        st.subheader("Select Source")
        source_filter = st.selectbox(
            "Choose a specific document (optional)",
            ["All Sources"] + sources
        )
        if source_filter == "All Sources":
            source_filter = None

    # Query input
    st.subheader("Ask a Question")
    question = st.text_area(
        "Enter your question",
        height=100,
        help="Ask a question about your uploaded documents"
    )

    # Query button
    if st.button("Get Answer", type="primary"):
        if not question:
            st.warning("Please enter a question.")
            return

        try:
            with st.spinner('Generating response...'):
                response = st.session_state.rag.query(
                    question=question,
                    source_filter=source_filter
                )

            if response:
                st.markdown("### Answer")
                st.write(response)

                # Add source context
                if source_filter:
                    st.caption(f"Response based on: {source_filter}")
                else:
                    st.caption("Response based on all available sources")
            else:
                st.warning("No response generated. Please try rephrasing your question.")

        except Exception as e:
            st.error(f"Error generating response: {str(e)}")

    # Document overview
    if sources:
        with st.expander("View Uploaded Documents"):
            st.markdown("### Available Documents")
            for source in sources:
                st.write(f"📄 {source}")
    else:
        st.info("No documents uploaded yet. Use the sidebar to upload PDFs.")

if __name__ == "__main__":
    main()