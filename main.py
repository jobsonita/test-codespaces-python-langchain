import os

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.openai import OpenAI


INDEX_NAME = "pdfs_embeddings"
embeddings = OpenAIEmbeddings()


def local_storage_exists():
    path = os.path.join(os.getcwd(), INDEX_NAME)
    return os.path.exists(path)


def load_vector_store():
    return FAISS.load_local(INDEX_NAME, embeddings)


def populate_vector_store():
    path = os.path.join(os.getcwd(), "pdfs", "ReAct Paper.pdf")
    path = os.path.abspath(path)

    loader = PyPDFLoader(path)
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    splitted_documents = text_splitter.split_documents(document)

    db = FAISS.from_documents(splitted_documents, embeddings)
    db.save_local(INDEX_NAME)

    return db


def search_vector_store(vectorstore: FAISS, text: str):
    results = vectorstore.similarity_search(text)
    print(results)
    print("")


def qa_chain(vectorstore: FAISS, query: str):
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
    )
    result = qa.run(query)
    print(result)


def main():
    vectorstore = load_vector_store() if local_storage_exists() else populate_vector_store()

    search_vector_store(vectorstore, "What is an agent?")

    qa_chain(vectorstore, "Give me the gist of ReAct in 3 sentences")


if __name__ == "__main__":
    main()
