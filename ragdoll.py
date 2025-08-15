from langchain_community.document_compressors import FlashrankRerank
from langchain_community.document_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import PyMuPDFParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain.tools import Tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_milvus import BM25BuiltInFunction, Milvus
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent, ToolNode, tools_condition
import os


class Ragdoll:
    def __init__(self, dir: str, temp: float, num_pred: int,
                 local: False, persist: False):
        self.dir = dir
        self.local = local
        self.persist = persist

        self.model = 'llama3.1:8b'
        self.temp = temp
        self.num_pred = num_pred

    def rerank_retrieve(self, query: str):
        """
            Rerank to boost similarity search results from in-memory store
        """
        rerank_retriever = ContextualCompressionRetriever(
            base_retriever=self.vector_store.as_retriever(search_kwargs={'k': 10}),
            base_compressor=FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=5),
        )

        return rerank_retriever.invoke(query)

    def hybrid_retrieve(self, query: str):
        """
            Semantic similarity x keyword search
        """
        return self.vector_store.similarity_search(
            query, k=5, ranker_type="weighted", ranker_params={"weights": [0.3, 0.7]}
        )

    def respond(self, prompt):
        """
            Outputs responses and records interaction history
        """
        config = {'configurable': {'thread_id': 'ragdoll'}}

        for step in self.graph.stream(
                {'messages': [{'role': 'user', 'content': prompt}]},
                stream_mode='values',
                config=config,
        ):
            step['messages'][-1].pretty_print()

            with open(f'doll_answer.txt', 'a') as outfile:
                outfile.write(f"{step['messages'][-1]}\n")

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
            - establishes connection to cloud vector store
              that supports hybrid semantic and keyword search
            OR
            - establishes in-memory vector store
            THEN
            - indexes docs as required
        """
        if not self.local or self.persist:
            self.vector_store = Milvus(
                connection_args={
                    'uri': os.environ['ZILLIZ_URI'],
                    'token': os.environ['ZILLIZ_TOKEN']
                },
                collection_name=self.dir,
                embedding_function=OllamaEmbeddings(model=self.model),
                builtin_function=BM25BuiltInFunction(),
                vector_field=["dense", "sparse"],
                consistency_level="Session",
                drop_old=False,
            )
        else:
            self.vector_store = InMemoryVectorStore(embedding=OllamaEmbeddings(model=self.model))

        if self.persist or self.local:
            self.index_pdfs()

    def index_pdfs(self):
        """
            - load *.pdfs from directory
            - split into 1000 len chunks
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

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=25)
        split_docs = text_splitter.split_documents(docs)

        self.vector_store.add_documents(documents=split_docs)