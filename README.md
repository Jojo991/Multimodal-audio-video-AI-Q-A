# Multimodal AI Q&A Bot with LangChain and OpenAI Whisper API

## Overview
This project demonstrates the development of a multimodal AI application that extracts meaningful insights from YouTube videos using advanced AI tools and frameworks. By leveraging LangChain, OpenAI's Whisper API, and embeddings, this system transcribes video audio, processes transcripts, and enables a powerful Q&A functionality.

## Features
- **Automated Video Transcription**: Converts YouTube video audio into structured text using OpenAI's Whisper API.
- **Intelligent Q&A Bot**: Creates an interactive question-answering experience using LangChain's RetrievalQA.
- **Advanced Vector Search**: Implements vector embeddings for fast, context-aware information retrieval.
- **Seamless Integration**: Utilizes `yt_dlp` for video downloading and `tiktoken` for efficient tokenization.

## Prerequisites
- Python 3.8 or higher
- OpenAI API Key
- Packages listed in `requirements.txt`


## Key Technologies
LangChain: Framework for building generative AI applications.

OpenAI Whisper API: Converts audio to text.

DocArray: Handles document storage and vectorization.

yt_dlp: Downloads YouTube videos efficiently.

tiktoken: Tokenizes text for embedding and processing.

## Future Enhancements
Integration with a vector database like Pinecone or Weaviate.

Support for multilingual transcription.

Expansion to handle multiple video sources simultaneously