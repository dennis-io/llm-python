# Import the necessary modules
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain import OpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAIEmbeddings object
embeddings = OpenAIEmbeddings()

# Load multiple documents from a directory and split them into chunks
loader = DirectoryLoader('news', glob='*.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=2500, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Create a Chroma index from the chunks and initialize a RetrievalQA object
docsearch = Chroma.from_documents(texts, embeddings)
qa = RetrievalQA.from_chain_type(
    llm = OpenAI(),
    chain_type = "stuff",
    retriever = docsearch.as_retriever()
)

# Define a function to run a query and print the answer
def query(q):
    print("Query: ", q)
    print("Answer: ", qa.run(q))

# Run some example queries
query("What is the meaning of life?")
query("What are chinas plans with renewable energy?")