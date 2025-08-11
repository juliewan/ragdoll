import argparse
from langchain_community.document_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import PyMuPDFParser
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_milvus import Milvus
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pymilvus import MilvusClient
from uuid import uuid4
import os


def calltool_or_respond(state):
    """
    based on user input
    - either call query tool to look up the topic
    - or answer directly
    """
    llm_with_tool = llm.bind_tools([query])
    response = llm_with_tool.invoke(state['messages'])
    return {'messages': [response]}

@tool
def query(query: str):

    """
    get k appropriate refs supporting queried topic(s)

    args:
        query (str): search term(s), topic(s)
    """
    return retriever.invoke(query)

def contextualized_response(state):

    # collate fresh prints off the vector exPRESS
    recent_messages = [message for message in reversed(state['messages']) if message.type == 'tool']
    tool_messages = recent_messages[::-1]
    context = "\n\n".join(doc.content for doc in tool_messages)
    system_message = (f"""
    
        You approach reading as an endeavor of unfettered learning and as toward
        magnanimous distillation.
        
        You refuse to gloss over the underlying technical structure, persisting
        until intimately conceptualizing each notion in its entirety.
        
        Reframe your thoughts into accessible language without losing precision
        or compromising the intent of the original material.
    
        Explain the following and intersperse applicable real-world examples.
        
        If there is anything that you do not know, simply state that you do not know.
        
        {context}
        
        """
    )
    conversation_messages = [
        message
        for message in state['messages']
        if message.type in ('human', 'system')
        or (message.type == 'ai' and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message)] + conversation_messages

    response = llm.invoke(prompt)
    return {'messages': [response]}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', help='question')
    parser.add_argument('dir', help='whether texts are in \'LOCAL:directory\' or \'CLOUD:collection\'?')
    parser.add_argument('-persist', help='\'Y\' to upload to cloud collection (optional)')
    parser.add_argument('-model', help='model with embedding and tool support from ollama.com/library (optional)')
    parser.add_argument('-temp', help='model temperature (optional)')
    parser.add_argument('-num_pred', help='max tokens to generate (optional)')
    args = parser.parse_args()

    store, dir = args.dir.split(':')
    model = args.model if args.model else "llama3.1:8b"
    temp = args.temp if args.temp else 0.25
    num_pred = args.num_pred if args.num_pred else 512

    llm = ChatOllama(
        model=model,
        temperature=temp,
        num_predict=num_pred,
    )

    if store == 'CLOUD' or (args.persist and args.persist == 'Y'):
        client = MilvusClient(
            uri=os.environ['ZILLIZ_URI'],
            token=os.environ['ZILLIZ_TOKEN']
        )
        collections = client.list_collections()

        vector_store = Milvus(
            connection_args={'uri': os.environ['ZILLIZ_URI'], 'token': os.environ['ZILLIZ_TOKEN']},
            embedding_function=OllamaEmbeddings(model=model),
            collection_name=dir,
            drop_old=False,  # True if drop existing collection
            index_params={'index_type': 'FLAT', 'metric_type': 'L2'},
            consistency_level='Strong',
        )
    else:
        vector_store = InMemoryVectorStore(embedding=OllamaEmbeddings(model=model))

    if store == 'LOCAL':
        # load all *.pdfs from folder
        loader = GenericLoader(
            blob_loader=FileSystemBlobLoader(
                path=args.dir.split(':')[1],
                glob="*.pdf",
            ),
            blob_parser=PyMuPDFParser(),
        )
        docs = loader.load()

        # split into 1000 char len chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=25)
        split_docs = text_splitter.split_documents(docs)

        vector_store.add_documents(documents=split_docs, ids=[str(uuid4()) for _ in range(len(split_docs))])

    if temp > 0.2:
        # if llm temp lil juiced,
        # fetch materials of greater diversity
        # useful if monolithic dataset to avoid overlapping info
        retriever = vector_store.as_retriever(
            search_type='mmr',
            search_kwargs={'k': 3,
                           'fetch_k': 10,  # num candidate text samples to pass to MMR
                           'lambda_mult': 1-temp}  # relevance diversity trade-off where 1 is min
        )
    else:
        # inflexibly pull refs in consensus to
        # hammer core concepts
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={'k': 2, 'score_threshold': 0.9}
        )

    """
    graph object of possb sequences:
        calltool_or_respond --> tools --> contextualized_response --> END
        calltool_or_respond --> END
        
    allows query_or_respond to respond directly if no tool call is made
    """

    graph_builder = StateGraph(MessagesState)

    tools = ToolNode([query])

    graph_builder.add_node(calltool_or_respond)
    graph_builder.add_node(tools)
    graph_builder.add_node(contextualized_response)

    graph_builder.set_entry_point('calltool_or_respond')
    graph_builder.add_conditional_edges(
        'calltool_or_respond',
        tools_condition,
        {END: END, 'tools': 'tools'},
    )
    graph_builder.add_edge('tools', 'contextualized_response')
    graph_builder.add_edge('contextualized_response', END)

    graph = graph_builder.compile()

    for step in graph.stream(
        {'messages': [{'role': 'user', 'content': args.prompt}]},
        stream_mode = 'values',
    ):
        step['messages'][-1].pretty_print()

        # record responses
        with open(f'doll_answer.txt', 'a') as outfile:
            outfile.write(f"{step['messages'][-1]}\n")
