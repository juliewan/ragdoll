# RAGdoll
Retrieval-augmented generation (RAG) doll that
- Indexes your provided *.pdf (text-only) to vector store (in-memory or cloud)
- Leverages
  - vector similarity + keyword match hybrid search (cloud)
  - reranked similarity search (in-memory)
- Grounds responses to your provided *.pdf files

## Example
```bash
Enter prompt (or 'q' to exit): What are paged optimizers in QLORA?

================================== Ai Message ==================================

Paged Optimizers is a technique used in QLORA to prevent memory spikes during gradient checkpointing. Here's a simplified explanation:
```

## Get Started

### Use [Ollama](https://github.com/ollama/ollama/blob/main/README.md#quickstart) for *free* local inferencing
Download model with embedding and tool support.
```bash
ollama pull llama3.1:8b
```
### Set up
```bash
pyenv virtualenv 3.12.5 <your-virtualenv-name>
```
```bash
pyenv activate <your-virtualenv-name>
```
```bash
pip install -r requirements.txt
```

### Use with in-memory vector store
Specify directory of *.pdf files for model context
- Indexed as in-memory vector store
- Retrieves most relevant excerpts to your prompt
- Responds to *contextualized* prompt
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