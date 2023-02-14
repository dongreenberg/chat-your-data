from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import SelfHostedEmbeddings
import runhouse as rh
import pickle

# Load Data
loader = UnstructuredFileLoader("state_of_the_union.txt")
raw_documents = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
documents = text_splitter.split_documents(raw_documents)


# Load Data to vectorstore
gpu = rh.cluster(name="rh-a10x", instance_type="A100:1", use_spot=False).save()
# gpu.restart_grpc_server(resync_rh=False, restart_ray=True)
model_reqs = ["local:./",
              # "git+https://github.com/dongreenberg/langchain.git@self_hosted",
              "pip:../langchain",
              "sentence_transformers",
              "torch"]
embeddings = SelfHostedEmbeddings(hardware=gpu, model_reqs=model_reqs)
vectorstore = FAISS.from_documents(documents, embeddings)


# Save vectorstore
with open("vectorstore.pkl", "wb") as f:
    pickle.dump(vectorstore, f)
