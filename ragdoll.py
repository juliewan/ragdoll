from langchain_community.document_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import PyMuPDFParser
from langchain_core.messages import SystemMessage
from langchain.tools import Tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_milvus import BM25BuiltInFunction, Milvus
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
import os


class Ragdoll:
    def __init__(self, dir: str, model: str, temp: float, num_pred: int,
                 local: False, persist: False):
        self.dir = dir
        self.model = model
        self.temp = temp
        self.num_pred = num_pred

        self.local = local
        self.persist = persist

    def calltool_or_respond(self, state):
        """
        - either calls tool to query vector store about the prompt
        - or proceeds to answer directly
        """
        llm_with_tool = self.llm.bind_tools([self.retrieve])
        response = llm_with_tool.invoke(state['messages'])
        return {'messages': [response]}

    def retrieve(self, query: str):
        return self.vector_store.similarity_search(query, k=3, distance_threshold=0.25)

    def hybrid_retrieve(self, query: str):
        return self.vector_store.similarity_search(
            query, k=3, ranker_type="weighted", ranker_params={"weights": [0.6, 0.4]}
        )

    def contextualized_response(self, state):
        """
            collate fresh prints off the vector exPRESS
            and fluff instructions with model context
        """
        recent_messages = [message for message in reversed(state['messages']) if message.type == 'tool']
        tool_messages = recent_messages[::-1]
        context = "\n\n".join(doc.content for doc in tool_messages)
        system_message = (f"""
            Explain the following with approachable real-world examples.
            Use accessible language without losing precision of the original material.

            If there is anything that you do not know, say \"I do not know.\"
            Do not make anything up.

            {context}
            """
                          )
        conversation_messages = [
            message for message in state['messages']
            if message.type in ('human', 'system')
               or (message.type == 'ai' and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message)] + conversation_messages

        response = self.llm.invoke(prompt)
        return {'messages': [response]}

    def build_llm_workflow(self):

        if not self.local or self.persist:
            retrieve_tool = self.hybrid_retrieve
        else:
            retrieve_tool = self.retrieve
        retrieve = Tool.from_function(
            func=retrieve_tool,
            name="retrieve",
            description="""
                retrieve appropriate reference materials to respond accordingly

                args:
                    query (str): search terminology
            """
        )
        tools = ToolNode([retrieve])

        workflow = StateGraph(state_schema=MessagesState)

        workflow.add_node(self.calltool_or_respond)
        workflow.add_node(tools)
        workflow.add_node(self.contextualized_response)

        workflow.set_entry_point('calltool_or_respond')
        workflow.add_conditional_edges(
            'calltool_or_respond',
            tools_condition,
            {END: END, 'tools': 'tools'},
        )
        workflow.add_edge('tools', 'contextualized_response')
        workflow.add_edge('contextualized_response', END)

        self.graph = workflow.compile(checkpointer=MemorySaver())

        self.llm = ChatOllama(
            model=self.model,
            temperature=self.temp,
            num_predict=self.num_pred,
        )

    def respond(self, prompt):
        config = {'configurable': {'thread_id': 'ragdoll'}}

        for step in self.graph.stream(
                {'messages': [{'role': 'user', 'content': prompt}]},
                stream_mode='values',
                config=config,
        ):
            step['messages'][-1].pretty_print()

            # record
            with open(f'doll_answer.txt', 'a') as outfile:
                outfile.write(f"{step['messages'][-1]}\n")

    def build_vector_store(self):
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
        # index *.pdfs from directory
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