import cohere
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from pypdf import PdfReader
import os
from typing import List, Dict
import uuid
import streamlit as st


class LocalRAG:
    def __init__(self, cohere_api_key: str, collection_name: str = "textbooks"):
        """
        Initialize local RAG system.
        
        Args:
            cohere_api_key: Your Cohere API key
            collection_name: Name for the Qdrant collection
        """
        # Initialize Cohere client
        self.co = cohere.Client(cohere_api_key)
        
        # Initialize local Qdrant client
        self.qdrant = QdrantClient(path="./qdrant_db")  # Stores data locally
        self.collection_name = collection_name
        
        # Create collection if it doesn't exist
        self._init_collection()
    
    def _init_collection(self):
        """Initialize Qdrant collection."""
        collections = self.qdrant.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            # Create collection for Cohere embeddings (1024 dimensions)
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=1024,
                    distance=Distance.COSINE
                )
            )
            print(f"Created new collection: {self.collection_name}")

    def _extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF file."""
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + " "
        return text.strip()

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into chunks with overlap."""
        words = text.split()
        chunks = []
        start = 0
        
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            
            if len(chunk.split()) >= 50:  # Minimum chunk size
                chunks.append(chunk)
            
            start += chunk_size - overlap
            
        return chunks

    def process_pdf(self, pdf_path: str, source_name: str = None) -> bool:
        """
        Process a PDF file and store it in the vector database.
        
        Args:
            pdf_path: Path to the PDF file
            source_name: Optional name to identify this source
        """
        try:
            # Use filename as source name if none provided
            if source_name is None:
                source_name = os.path.basename(pdf_path)
            
            print(f"Processing {source_name}...")
            
            # Extract and chunk text
            text = self._extract_text(pdf_path)
            chunks = self._chunk_text(text)
            
            if not chunks:
                print("No valid text chunks extracted")
                return False
            
            # Process chunks in batches
            batch_size = 96  # Cohere's recommended batch size
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                # Generate embeddings
                embeddings = self.co.embed(
                    texts=batch,
                    model="embed-english-v3.0",
                    input_type="search_document"
                ).embeddings
                
                # Prepare points for Qdrant
                points = []
                for chunk, embedding in zip(batch, embeddings):
                    point_id = str(uuid.uuid4())
                    points.append(PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "text": chunk,
                            "source": source_name
                        }
                    ))
                
                # Upload to Qdrant
                self.qdrant.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                
                print(f"Processed batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
            
            print(f"Successfully processed {source_name}")
            return True
            
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            return False

    def query(self, question: str, source_filter: str = None) -> str:
        """
        Query the RAG system.
        
        Args:
            question: User's question
            source_filter: Optional source name to filter results
        """
        try:
            # Generate query embedding
            query_embedding = self.co.embed(
                texts=[question],
                model="embed-english-v3.0",
                input_type="search_query"
            ).embeddings[0]
            
            # Search in Qdrant
            search_filter = None
            if source_filter:
                search_filter = {
                    "must": [
                        {"key": "source", "match": {"value": source_filter}}
                    ]
                }
            
            # In your query method, increase the number of retrieved chunks
            search_results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=8,  # Increase from 3 to 5 for more context
                query_filter=search_filter
            )
            
            if not search_results:
                return "No relevant information found."
            
            # Prepare context from search results
            context = "\n\n".join([hit.payload["text"] for hit in search_results])
            
            # Generate response using Command-R+
            response = self.co.chat(
                message=question,
                model="command-r-plus-08-2024",
                preamble = """You are an expert professor in satellite navigation, guidance, and control systems. 
                When answering questions:

                1. Always start with a high-level overview of the concept
                2. Give detailed, thorough explanations with examples
                3. Use mathematical expressions when relevant
                4. Break down complex ideas into digestible parts
                5. ALWAYS aim for comprehensive, lecture-style responses
                6. Explicitly connect concepts to real-world satellite applications
                7. Minimum response length should be several paragraphs
                8. Include specific technical terminology with explanations
                9. Reference related concepts to build a broader understanding
                10. Give practical examples and applications of the concepts even if they are out of context

                If the context doesn't contain all needed information, explain what you can from the available context 
                and clearly indicate what additional aspects would typically be relevant for a complete understanding.

                Remember: Your goal is to provide lecture-quality, comprehensive explanations that a graduate student 
                would find valuable. Never be brief - always expand and elaborate.""",
                
                documents=[{"text": context}],
                temperature=0.5,
                # connectors=[{"id": "follow-up"}],
                # suggest_follow_up_questions=True,

                # Add citation to show where information comes from
                citation_quality="accurate"
                )
            
            return response.text
            
        except Exception as e:
            print(f"Error querying RAG system: {str(e)}")
            return None

    def list_sources(self) -> List[str]:
        """List all available source documents."""
        response = self.qdrant.scroll(
            collection_name=self.collection_name,
            scroll_filter=None,
            limit=100,
            with_payload=["source"],
            with_vectors=False
        )
        
        sources = set()
        for point in response[0]:
            sources.add(point.payload["source"])
        
        return list(sources)


# Example usage
if __name__ == "__main__":
    # Initialize the system
    rag = LocalRAG(cohere_api_key="24lN212hrjQAecMqJPfD01wirgAo0UZ7eGNQo2iB")
    
    # Process some PDFs
    pdf_directory = "pdf"
    for filename in os.listdir(pdf_directory):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_directory, filename)
            rag.process_pdf(pdf_path)
    
    # List available sources
    sources = rag.list_sources()
    # print("\nAvailable sources:", sources)
    # Add this to your main() function after getting sources
    st.write("Number of documents:", len(sources))
    print("Available sources:", sources)
    
    # Ask questions
    while True:
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
        
        response = rag.query(question)
        print("\nResponse:", response)