from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document


def generate_qa_pairs(f1, f2, f3):
    qa_pairs = []
    with open(f1, "r") as file1, open(f2, "r") as file2, open(f3, "r") as file3:
        for q, c, a in zip(file1, file2, file3):
            qa_pairs.append((q.strip(), c.strip(), a.strip()))
    return qa_pairs


qa_pairs = generate_qa_pairs(
    "./db/train/train.question",
    "./db/train/train.code",
    "./db/train/train.answer")
docs = [f"Question: {q} Code: {c}" for q, c, a in qa_pairs]
answers = [f"Answer: {a}" for _, __, a in qa_pairs]


client = QdrantClient(path="./qdrant")

model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2")  # 384-мерный вектор
client.create_collection(
    collection_name="disk_collection",
    vectors_config={"size": 384, "distance": "Cosine"}
)
print("Empty collection created")

print("Embedding...")
db = QdrantVectorStore(
    client=client,
    embedding=model,
    collection_name="disk_collection"
)
documents = [
    Document(
        page_content=i,
        metadata={
            "lang": "python"}) for i in docs]


print("Collection of vectors is created")

print("Add documents")

db.add_documents(documents=documents, ids=[id for id in range(len(docs))])
print("documents added")
