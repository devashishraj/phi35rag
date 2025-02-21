from sentence_transformers import SentenceTransformer

model = SentenceTransformer("jinaai/jina-embeddings-v3",trust_remote_code=True)

saveModelTo="/app/jinv3"
model.save(saveModelTo,safe_serialization=True)
