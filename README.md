## Ask Doll
A study buddy that reads books and articles that you can quiz!

### Use [Ollama](https://github.com/ollama/ollama/blob/main/README.md#quickstart) for *free* local inferencing
```bash
ollama pull llama3.1:8b
```
### Set up
```bash
pyenv virtualenv 3.12.5 <environment-name>
```
```bash
pyenv activate <environment-name>
```
```bash
pip install -r requirements.txt
```

### Running with in-memory vector store
```bash
python3 ask_doll.py "What is multi-head and self-attention?" LOCAL:research_papers
```
### Specify Ollama model (that has embedding and tool support)
```bash
python3 ask_doll.py "What is multi-head and self-attention?" LOCAL:research_papers -model mistral:7b
```
### Tweak model temperature and max token generation
```bash
python3 ask_doll.py "What is multi-head and self-attention?" LOCAL:research_papers -temp 0.88 -num_pred 1024
```

### Use [Zilliz](https://zilliz.com/pricing)'s *free* cloud vector store
- 5 GB storage
- Up to 5 collections

```bash
export ZILLIZ_URI=
```
```bash
export ZILLIZ_TOKEN=
```

### Upload texts to cloud collection
(The collection name will be the directory name)
```bash
python3 ask_doll.py "Question?" LOCAL:research_papers -persist Y
```
### Query from uploaded texts
```bash
python3 ask_doll.py "Question?" CLOUD:research_papers
```
