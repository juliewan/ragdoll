from __future__ import annotations
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.document_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import PyMuPDFParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain.tools import Tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_experimental.text_splitter import SemanticChunker
from flashrank import Ranker
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_milvus import BM25BuiltInFunction, Milvus
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent, ToolNode, tools_condition
from datetime import datetime
import json
import os
import re


class Ragdoll:
    def __init__(self, dir: str, temp: float, num_pred: int,
                 local: False, persist: False):
        self.dir = dir
        self.local = local
        self.persist = persist

        self.model = 'llama3.1:8b'
        self.temp = temp
        self.num_pred = num_pred

        # Dedicated retrieval embeddings (384-d, ONNX): better recall and ~10x
        # smaller vectors than reusing the generative chat model for embeddings.
        # Pin cache_dir so the model isn't re-downloaded when macOS purges its
        # default temp cache (/var/folders/.../T/fastembed_cache).
        self.embeddings = FastEmbedEmbeddings(
            model_name='BAAI/bge-small-en-v1.5',
            cache_dir=os.path.expanduser('~/.cache/fastembed'),
        )

    def rerank_retrieve(self, query: str):
        """
            Cross-Encoder rerank to boost similarity search
        """
        FlashrankRerank.model_rebuild()

        # Build the reranker once and pin its cache_dir. FlashRank's Ranker
        # defaults to /tmp, which macOS purges (forcing a re-download every run);
        # the wrapper doesn't expose cache_dir, so we pass our own Ranker client.
        if not hasattr(self, 'reranker'):
            self.reranker = FlashrankRerank(
                client=Ranker(
                    model_name="ms-marco-MiniLM-L-12-v2",
                    cache_dir=os.path.expanduser('~/.cache/flashrank'),
                ),
                top_n=3,
            )

        rerank_retriever = ContextualCompressionRetriever(
            base_retriever=self.vector_store.as_retriever(search_kwargs={'k': 10}),
            base_compressor=self.reranker,
        )

        return rerank_retriever.invoke(query)

    def hybrid_retrieve(self, query: str):
        """
            AKA "Fusion Retrieval"
            semantic similarity x keyword search
            with weighted reranking
        """
        return self.vector_store.similarity_search(
            query, k=3, ranker_type="weighted", ranker_params={"weights": [0.3, 0.7]}
        )

    @staticmethod
    def _slugify(text: str, max_words: int = 8, max_len: int = 60) -> str:
        """
            Filesystem-safe topic slug derived from the first prompt.
        """
        words = re.sub(r'[^a-z0-9\s-]', '', text.lower()).split()
        return '-'.join(words[:max_words])[:max_len].strip('-') or 'chat'

    @staticmethod
    def _extract_sources(tool_content) -> list[str]:
        """
            Pull source filenames out of a retrieve tool result for citation.
        """
        paths = re.findall(r"'source': '([^']+)'", str(tool_content))
        return [os.path.basename(p) for p in paths]

    def _log(self, role: str, content: str, sources: list[str] = None):
        """
            Append one conversation turn as a JSON line to the session log.
        """
        record = {
            'ts': datetime.now().isoformat(timespec='seconds'),
            'role': role,
            'content': content,
        }
        if sources:
            record['sources'] = sources

        with open(self.log_path, 'a') as outfile:
            outfile.write(json.dumps(record, ensure_ascii=False) + '\n')

    def respond(self, prompt):
        """
            Streams the response to the console and appends the turn to a
            per-session JSONL log at logs/{date}_{topic}.jsonl. Only the user
            prompt and the grounded answer are recorded (with the source files
            retrieved); intermediate tool-call chatter stays out of the log.
        """
        config = {'configurable': {'thread_id': 'ragdoll'}}

        if not hasattr(self, 'log_path'):
            os.makedirs('logs', exist_ok=True)
            name = f"{datetime.now():%Y-%m-%d}_{self._slugify(prompt)}.jsonl"
            self.log_path = os.path.join('logs', name)

        self._log(role='user', content=prompt)

        answer, sources = '', []
        for step in self.graph.stream(
                {'messages': [{'role': 'user', 'content': prompt}]},
                stream_mode='values',
                config=config,
        ):
            message = step['messages'][-1]
            message.pretty_print()

            if message.type == 'tool':
                sources.extend(self._extract_sources(message.content))
            elif message.type == 'ai' and not message.tool_calls:
                answer = message.content

        self._log(role='assistant', content=answer, sources=sorted(set(sources)))

    def build_react_graph(self):
        """
            ReAct (Reasoning x Acting) agent that calls tools until stopping condition
        """
        retrieve_tool = self.hybrid_retrieve \
                        if not self.local or self.persist \
                        else self.rerank_retrieve

        retrieve = Tool.from_function(
            func=retrieve_tool,
            name="retrieve",
            description="""
                Retrieve information regarding specified topic(s).

                Args:
                    query (str): search keywords, terminology, phrases

                Returns:
                    documents with relevant page_content and metadata
            """
        )
        self.graph = create_react_agent(
            model=ChatOllama(
                model=self.model,
                temperature=self.temp,
                num_predict=self.num_pred,
            ),
            tools=[retrieve],
            prompt="""
                Retrieve information by calling tool 'retrieve' and providing search term(s)
                to obtain context to ground your response, then respond accordingly.
                
                Frame with accessible language without losing bedrock of the reference material.
                
                When appropriate, offer approachable real-world examples.

                If there is anything that you do not know, say \"I do not know.\"
                
                Do not infer. Do not make anything up.
            """,
            checkpointer=MemorySaver(),
        )

    def build_vector_store(self):
        """
            SPECIFIES EITHER
            - cloud collection with multi-vector field
              that supports semantic and keyword search
              and weighted re-ranking
            - in-memory vector store

            THEN
            - indexes texts as required
        """
        if not self.local or self.persist:
            self.vector_store = Milvus(
                connection_args={
                    'uri': os.environ['ZILLIZ_URI'],
                    'token': os.environ['ZILLIZ_TOKEN']
                },
                collection_name=self.dir,
                embedding_function=self.embeddings,
                builtin_function=BM25BuiltInFunction(),
                vector_field=["dense", "sparse"],
                consistency_level="Session",
                drop_old=False,
            )
        else:
            self.vector_store = InMemoryVectorStore(embedding=self.embeddings)

        if self.persist or self.local:
            self.index_pdfs()

    def index_pdfs(self):
        """
            - load *.pdfs from directory
            - split into semantically delineated chunks
            - index to vector store
        """
        loader = GenericLoader(
            blob_loader=FileSystemBlobLoader(
                path=self.dir,
                glob="*.pdf",
            ),
            blob_parser=PyMuPDFParser(),
        )
        docs = loader.load()

        text_splitter = SemanticChunker(self.embeddings,
                                        breakpoint_threshold_type="percentile")
        split_docs = text_splitter.split_documents(docs)

        self.vector_store.add_documents(documents=split_docs)