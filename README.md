# RAGdoll
Retrieval augmented generation (RAG)doll that
- indexes your provided *.pdf (text-only) to vector store (in-memory or cloud)
- leverages vector similarity + keyword matching hybrid search (cloud only)
- grounds responses to your provided *.pdf files

## Example
```bash
Enter prompt (or 'q' to exit): What are paged optimizers in QLORA?

================================== Ai Message ==================================

Paged Optimizers is a technique used in QLORA to prevent memory spikes during gradient checkpointing. Here's a simplified explanation:
```

## Get Started

### Use [Ollama](https://github.com/ollama/ollama/blob/main/README.md#quickstart) for *free* local inferencing
Download a text model with embedding and tool support that can run on your system
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

### Prompt with in-memory vector store
Specify directory of *.pdf files for model reference
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
The directory name will become the vector store collection name
```bash
python main.py -d <your-directory-name> --local --persist
```
### 2. Query from cloud vector store
No indexing, should feel responsive
```bash
python main.py -d <your-collection-name>
```
### Tweak LLM temperature and max token generation
```bash
python main.py -d <your-collection-name> --temp 0.88 --num_pred 1024
```
### Swap Ollama model with another (that has embedding and tool support)
```bash
python main.py -d <your-collection-name> --model mistral:7b
```
