# Retrieval-Augmented Generation (RAG) Pipeline for Technical PDFs

## Project Objective

Build a Retrieval-Augmented Generation (RAG) system that extracts information from technical PDF documents and answers questions using semantic search and large language models while minimizing hallucination.

The system combines document retrieval with AI-generated responses to ensure answers are grounded in external knowledge.

## Learning Goals

- Understand the full RAG architecture (ingestion → embeddings → retrieval → generation)
- Practice text preprocessing and chunking strategies
- Work with vector databases (ChromaDB)
- Generate embeddings using OpenAI models
- Implement semantic search
- Evaluate hallucination control in LLM systems
- Build a modular AI pipeline

## Project Structure

- `notebooks/` – Main Jupyter notebook implementing the RAG pipeline
- `data/` – Technical PDF documents used as the knowledge base
- `data/chroma_db/` – Persistent vector database storage (Chroma)
- `src/` – Core implementation of the RAG pipeline (extraction, chunking, embedding, retrieval)

## Setup

This project requires an OpenAI API key to generate embeddings and model responses.

Before running the notebook, set your API key as an environment variable:

- On Windows (PowerShell):
setx OPENAI_API_KEY "your_api_key_here"

- On macOS / Linux:
export OPENAI_API_KEY="your_api_key_here"

## Dataset

- Collection of technical PDF documents
- Topics related to DevOps, DORA metrics, and software engineering
- Documents are processed, cleaned, chunked, and embedded

The technical PDF documents used in this project were generated from publicly available Wikipedia articles.

Wikipedia content is licensed under the Creative Commons Attribution-ShareAlike License (CC BY-SA), which allows reuse for educational and research purposes with proper attribution.

Source: https://www.wikipedia.org/

## Notebook

The main implementation and evaluation of the RAG pipeline are documented in the Jupyter notebook located in the `notebooks/` directory.

The notebook includes:
- Step-by-step implementation of the pipeline
- Detailed explanations of each component
- Hallucination testing experiments
- Final results and conclusions

## RAG Pipeline Overview

### 1. Data Extraction
- Extract text from PDFs using `pdfplumber`
- Split text into paragraphs
- Clean unwanted elements (URLs, citations, extra whitespace)

### 2. Chunking
- Split text into sentence-based chunks
- Apply a maximum word limit per chunk
- Add overlap between chunks to preserve context

### 3. Embedding Generation
- Use OpenAI embedding model (`text-embedding-3-small`)
- Convert each chunk into vector representations
- Store embeddings in a persistent Chroma vector database

### 4. Retrieval
- Convert user query into embedding
- Perform semantic similarity search
- Retrieve the most relevant chunks

### 5. Answer Generation
- Build a context-based prompt
- Instruct the model to answer only using retrieved documents
- Generate final response using GPT model

## Evaluation

The system was tested using multiple scenarios:

- Direct knowledge questions (e.g., DORA metrics)
- Partial information scenarios
- False assumption tests
- Hallucination tests (questions outside the documents)

Evaluation focused on:
- Context grounding
- Retrieval quality
- Hallucination prevention
- Model behavior under constraints

## Conclusions

- The RAG pipeline successfully retrieves relevant information from technical PDFs.
- The system produces accurate answers when information exists in the documents.
- When information is not available, the model avoids hallucination and responds appropriately.
- Combining vector search with large language models improves factual reliability and domain-specific performance.

This project demonstrates how Retrieval-Augmented Generation enhances LLMs by grounding them in external knowledge sources.

## Next Steps / Improvements

- Streaming responses – Enable streaming from the language model so long answers are displayed gradually instead of waiting for the full response.
- Top-k retrieval tuning – Experiment with different values of `top_k` to improve retrieval performance, especially for complex or multi-part questions.
- Enhanced metadata usage – Include additional metadata (such as source document name, paragraph index, or chunk ID) directly in the prompt to improve traceability and transparency.
- Embedding optimization and caching – Store embeddings permanently and avoid recomputing them on every execution to reduce API usage and improve efficiency.
- Advanced retrieval strategies – Implement techniques such as reranking, hybrid search (keyword + semantic), or semantic chunking to further improve retrieval quality.