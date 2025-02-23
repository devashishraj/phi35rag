from sentence_transformers import SentenceTransformer
import torch
saveModelTo="/app/jinv3"

modelKwargs={"torch_dtype":torch.bfloat16}
model = SentenceTransformer(
        "jinaai/jina-embeddings-v3",
        trust_remote_code=True,
        model_kwargs=modelKwargs,
    )

model.save(saveModelTo,safe_serialization=True)


