from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import json
from fastembed import TextEmbedding
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")


print("starting vector store ingestion script...")

emb_dim = 384

# Local file-backed (persists between runs)
client = QdrantClient(path="./qdrant_data")

# In-memory (wiped on exit)
# client = QdrantClient(":memory:")

# Create a collection (like a table, but for vectors)

existing = [c.name for c in client.get_collections().collections]
if "components" not in existing:
    client.create_collection(
        collection_name="components",
        vectors_config=VectorParams(size=emb_dim, distance=Distance.COSINE),
    )
    print("Collection created")
else:
    print("Collection already exists, skipping creation")





# load data
json1_file = open('data/components.json')
json1_str = json1_file.read()
json1_data = json.loads(json1_str)

# collect descriptions
documents = []
for component in json1_data['components']:
    documents.append(component['description'])

# embed descriptions
embedding_model = TextEmbedding()
embeddings_generator = embedding_model.embed(documents)
embeddings_list = list(embeddings_generator)
assert len(embeddings_list[0]) == emb_dim
print(f"Embeddings are of size: {len(embeddings_list[0])}")


# prepare collection inputs
points_list = []
i = 0

for component, vector in zip(json1_data['components'],embeddings_list):
    entry = PointStruct(id = i, vector = vector,
                        payload = {"id": component["id"], "name": component['name'],
                                   "subsystem": component["subsystem"], "description": component["description"] 
                                   })
    points_list.append(entry)
    i +=1
    

# add to collection
client.upsert(
    collection_name="components",
    points= points_list
)
print("Entries successfully added to Vector Store. Now running test query...")

# test query
query = "gasket for oil pan to transmission"
query_vector = list(embedding_model.embed(query))[0]

results = client.query_points(
    collection_name="components",
    query=query_vector,   
    limit=5,
)

print("Results of the test query:\n")
for r in results.points:
    print(r.id, r.score, r.payload)


client.close()
print("Client closed cleanly.")
