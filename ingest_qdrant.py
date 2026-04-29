from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    SparseVectorParams, SparseIndexParams, SparseVector,
)
from fastembed import TextEmbedding, SparseTextEmbedding
import json
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
    vectors_config={
        "dense": VectorParams(size=emb_dim, distance=Distance.COSINE),
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams(
            index=SparseIndexParams(on_disk=False)
        ),
    },
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
sparse_model = SparseTextEmbedding("Qdrant/bm25")

dense_vecs  = list(embedding_model.embed(documents))
sparse_vecs = list(sparse_model.embed(documents))
points_list = []

for i, (component, dvec, svec) in enumerate(
    zip(json1_data['components'], dense_vecs, sparse_vecs)
):
    entry = PointStruct(
        id=i,
        vector={
            "dense": dvec.tolist(),
            "sparse": SparseVector(
                indices=svec.indices.tolist(),
                values=svec.values.tolist(),
            ),
        },
        payload={
            "id":          component["id"],
            "name":        component["name"],
            "subsystem":   component["subsystem"],
            "description": component["description"],
        }
    )
    points_list.append(entry)

    

# add to collection
client.upsert(
    collection_name="components",
    points= points_list
)
print("Entries successfully added to Vector Store. Now running test query...")

# test query
query = "gasket for oil pan to transmission"
dense_query_vec  = list(embedding_model.embed(query))[0]
sparse_query_vec = list(sparse_model.embed(query))[0]

# dense test
dense_results = client.query_points(
    collection_name="components",
    query=dense_query_vec.tolist(),
    using="dense",
    limit=5,
)

print("\nDense (semantic) results:")
for r in dense_results.points:
    print(f"  {r.score:.3f}  {r.payload['name']}")

# sparse test
sparse_results = client.query_points(
    collection_name="components",
    query=SparseVector(
        indices=sparse_query_vec.indices.tolist(),
        values=sparse_query_vec.values.tolist(),
    ),
    using="sparse",
    limit=5,
)

print("\nSparse (BM25) results:")
for r in sparse_results.points:
    print(f"  {r.score:.3f}  {r.payload['name']}")
print("Client closed cleanly.")
