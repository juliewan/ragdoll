# RAGdoll
Retrieval-augmented generation (RAG) doll that
- Indexes your *.pdfs (text-only, no img yet) to in-memory/cloud vector store as semantically coherent chunks
  - deriving semantically coherent chunks from your texts is a resource intensive step, try to set up cloud ASAP to persist embeddings
- Retrieves via
  - **hybrid/fusion (cloud)**
    - vector semantic similarity
    - lexical search (keyword match)
    - weighted reranking
  - **relevance reranking (in-memory)**
    - vector semantic similarity
    - cross-encoder reranking
- Grounds responses to your provided *.pdf context for bespoke feel without fine-tuning

## Example
```bash
Enter prompt (or 'q' to exit): ELI5 paged optimizers

================================== Ai Message ==================================

Paged optimizers are like a special kind of memory manager for computers...
```

## Get Started

### Use [Ollama](https://github.com/ollama/ollama/blob/main/README.md#quickstart) served model for *free* local inferencing
Download model with tool support:
```bash
ollama pull llama3.1:8b
```
The dedicated embeddings model `BAAI/bge-small-en-v1.5` (384-d, ONNX) for indexing, semantic chunking, and retrieving will download automatically via [fastembed](https://github.com/qdrant/fastembed) at first run.
### Set up
Python 3.11+:
```bash
python -m venv venv
```
```bash
source venv/bin/activate        # Windows: venv\Scripts\activate
```
```bash
pip install -r requirements.txt
```

### Use with in-memory vector store
Specify directory of *.pdf files to interactively query
- Indexed as in-memory vector store
- Reranks most semantically relevant excerpts to your prompt
- Interpolates prompt with retrieved context
```bash
python main.py -d research_papers --local
```
## Cloud
### Use [Zilliz](https://zilliz.com/pricing)'s *free* tier
- 5 GB storage
- Up to 5 collections

```bash
export ZILLIZ_URI=<your-zilliz-uri>
```
```bash
export ZILLIZ_TOKEN=<your-zilliz-token>
```

### 1. Upload texts to cloud vector store
The collection name will be the same as the directory name.
```bash
python main.py -d <your-directory-name> --local --persist
```
### 2. Query from cloud vector store
Et voila. No indexing step, should feel responsive.
```bash
python main.py -d <your-collection-name>
```
### Tweak temperature and max token output
```bash
python main.py -d <your-collection-name> --temp 0.88 --num_pred 1024
```