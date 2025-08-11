# ragdoll
Retrieval augmented generation (RAG) doll that
- indexes your provided *.pdf (text-only) to vector store (in-memory or cloud)
- grounds responses to your provided *.pdf files

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
- Directory will be indexed as in-memory vector store
- Ragdoll will retrieve most relevant texts to your prompt
- Your prompt will be contextualized with retrieved supporting texts
- Ragdoll will respond to the contextualized prompt
```bash
python3 ask_doll.py "What is multi-head and self-attention?" LOCAL:research_papers
```
### Swap Ollama model to another (that has embedding and tool support)
```bash
python3 ask_doll.py <your-prompt> LOCAL:<your-directory-name> -model mistral:7b
```
### Tweak model temperature and max token generation
```bash
python3 ask_doll.py <your-prompt> LOCAL:<your-directory-name> -temp 0.88 -num_pred 1024
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

### Upload texts to cloud vector store
The directory name will become the vector store collection name
```bash
python3 ask_doll.py <your-prompt> LOCAL:research_papers -persist Y
```
### Query from cloud vector store
No indexing, Ragdoll should feel responsive
```bash
python3 ask_doll.py <your-prompt> CLOUD:research_papers
```
