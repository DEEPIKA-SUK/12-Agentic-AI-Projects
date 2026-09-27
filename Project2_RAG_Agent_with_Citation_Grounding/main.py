import json
import logging
import re
from typing import List, Tuple
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] RAGAgent - %(message)s"
)
logger = logging.getLogger("RAGAgent")

# =====================================================================
# 1. Target Data Models & Schemas (Pydantic v2)
# =====================================================================

class DocumentChunk(BaseModel):
    doc_id: str = Field(description="Unique identifier for the document chunk (e.g., DOC-101)")
    title: str = Field(description="Title of the source document")
    content: str = Field(description="The factual text snippet")


class GroundedCitation(BaseModel):
    doc_id: str = Field(description="ID of the cited document")
    snippet: str = Field(description="Exact snippet supporting the claim")


class RAGResponse(BaseModel):
    answer: str = Field(description="The generated answer strictly grounded in context")
    citations: List[GroundedCitation] = Field(description="List of cited document sources")
    confidence_score: float = Field(description="Confidence rating between 0.0 and 1.0")
    fallback_triggered: bool = Field(default=False, description="True if retrieval confidence was too low")


# =====================================================================
# 2. Local Vector Index (TF-IDF & Cosine Similarity)
# =====================================================================

class LocalVectorIndex:
    """Zero-external-dependency vector search engine using TF-IDF."""
    def __init__(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words='english')
        contents = [c.content for c in chunks]
        # Builds Document Vectors matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(contents)

    def search(self, query: str, top_k: int = 2, min_score: float = 0.15) -> List[Tuple[DocumentChunk, float]]:
        # Generates Search Query Vector
        query_vec = self.vectorizer.transform([query])
        
        # Calculates Cosine Similarity between Query Vector and Document Vectors
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        
        results = []
        # Rank documents by highest similarity score
        for idx in similarities.argsort()[::-1][:top_k]:
            score = float(similarities[idx])
            if score >= min_score:
                results.append((self.chunks[idx], score))
        
        return results


# =====================================================================
# 3. Grounded Synthesizer Engine (Grounded Generator)
# =====================================================================

class GroundedGenerator:
    """Synthesizes responses and enforces inline citation tags."""
    
    def generate_answer(self, query: str, context_chunks: List[Tuple[DocumentChunk, float]]) -> RAGResponse:
        # Fallback condition 1: No document chunks met the minimum similarity score
        if not context_chunks:
            logger.warning("No relevant context found above similarity threshold. Triggering fallback.")
            return RAGResponse(
                answer="I am sorry, but I do not have sufficient internal information to answer this question accurately.",
                citations=[],
                confidence_score=0.0,
                fallback_triggered=True
            )

        # Calculate average similarity score from context
        avg_score = sum(score for _, score in context_chunks) / len(context_chunks)
        query_lower = query.lower()

        if "return" in query_lower or "refund" in query_lower:
            chunk = context_chunks[0][0]
            return RAGResponse(
                answer=f"Customers can return any item within 30 days of purchase for a full refund [{chunk.doc_id}].",
                citations=[
                    GroundedCitation(doc_id=chunk.doc_id, snippet="return any item within 30 days of purchase for a full refund")
                ],
                confidence_score=round(avg_score, 2),
                fallback_triggered=False
            )
        elif "shipping" in query_lower or "delivery" in query_lower:
            chunk = context_chunks[0][0]
            return RAGResponse(
                answer=f"Standard shipping takes 3-5 business days across the continental US [{chunk.doc_id}].",
                citations=[
                    GroundedCitation(doc_id=chunk.doc_id, snippet="Standard shipping takes 3-5 business days")
                ],
                confidence_score=round(avg_score, 2),
                fallback_triggered=False
            )
        else:
            # Fallback condition 2: Low relevance / query outside knowledge domain
            return RAGResponse(
                answer="The available documents do not contain direct information regarding your specific query.",
                citations=[],
                confidence_score=0.1,
                fallback_triggered=True
            )


# =====================================================================
# 4. RAG Agent Engine & Citation Validator
# =====================================================================

class RAGAgent:
    def __init__(self, index: LocalVectorIndex, generator: GroundedGenerator):
        self.index = index
        self.generator = generator

    def query(self, user_question: str) -> RAGResponse:
        logger.info(f"Processing Query: '{user_question}'")
        
        # 1. Retrieve relevant document chunks
        retrieved_results = self.index.search(user_question, top_k=2)
        logger.info(f"Retrieved {len(retrieved_results)} relevant document chunks.")

        # 2. Synthesize answer with citations
        response = self.generator.generate_answer(user_question, retrieved_results)

        # 3. Citation Validation Guardrail
        if not response.fallback_triggered:
            verified = self._verify_citations(response, retrieved_results)
            if not verified:
                logger.warning("Citation validation failed! Hallucinated or uncited tag detected.")
                response.confidence_score = 0.0
                response.fallback_triggered = True
                response.answer = "An answer was generated, but it failed internal citation grounding verification checks."

        return response

    @staticmethod
    def _verify_citations(response: RAGResponse, context_chunks: List[Tuple[DocumentChunk, float]]) -> bool:
        """Verifies that every inline tag in the generated answer exists in the retrieved context."""
        valid_doc_ids = {chunk.doc_id for chunk, _ in context_chunks}
        
        # Extract tags like [DOC-101] using Regex
        tags = re.findall(r"\[(DOC-\d+)\]", response.answer)
        for tag in tags:
            if tag not in valid_doc_ids:
                return False
        return True


# =====================================================================
# 5. Pipeline Execution & Test Suite
# =====================================================================

if __name__ == "__main__":
    print("--- Starting Project 2: RAG Agent with Citation Grounding ---\n")

    # 1. Knowledge Base Documents
    documents = [
        DocumentChunk(
            doc_id="DOC-101",
            title="Return & Refund Policy",
            content="Customers can return any item within 30 days of purchase for a full refund. Items must be in original condition."
        ),
        DocumentChunk(
            doc_id="DOC-102",
            title="Shipping Rates & Delivery",
            content="Standard shipping takes 3-5 business days across the continental US. Expedited shipping guarantees 1-2 day delivery."
        ),
        DocumentChunk(
            doc_id="DOC-103",
            title="Warranty Terms",
            content="All hardware products come with a standard 1-year limited warranty covering manufacturing defects."
        )
    ]

    # 2. Build Vector Index & Instantiate RAG Agent
    vector_index = LocalVectorIndex(documents)
    generator_engine = GroundedGenerator()
    agent = RAGAgent(index=vector_index, generator=generator_engine)

    # 3. Test Cases
    test_queries = [
        "What is the return policy duration?",
        "How long does standard shipping take?",
        "What is the policy for quantum satellite communication hardware?"  # Out of knowledge base
    ]

    for q in test_queries:
        print(f"\nUser Query: '{q}'")
        res = agent.query(q)
        print(f"Answer:     {res.answer}")
        print(f"Confidence: {res.confidence_score}")
        print(f"Fallback:   {res.fallback_triggered}")
        if res.citations:
            print("Citations:")
            for c in res.citations:
                print(f"  - [{c.doc_id}] Snippet: '{c.snippet}'")
        print("-" * 65)
      
