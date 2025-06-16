import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph
from typing import List, TypedDict
from langchain_core.documents import Document

def initialize_rag_pipeline():
    # Initialize components
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("train-info")
    vector_store = PineconeVectorStore(embedding=embeddings, index=index)

    # Define prompt template
    prompt_template = PromptTemplate.from_template('''
You are Railbuddy a respectful sarcastic assistant for question-answering tasks for train schedules and Information.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know. Use five sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:''')

    # Define state
    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str

    # Define graph nodes
    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt_template.format(question=state["question"], context=docs_content)
        response = llm.invoke(messages)
        return {"answer": response.content}

    # Build graph
    graph_builder = StateGraph(State)
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("generate", generate)
    graph_builder.set_entry_point("retrieve")
    graph_builder.add_edge("retrieve", "generate")
    return graph_builder.compile()
