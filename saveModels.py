from sentence_transformers import SentenceTransformer
import torch
saveModelTo="/app/jinv3"
modelCachePath="/app/jinv3/modelCache"
modelKwargs={"torch_dtype":torch.bfloat16}
model = SentenceTransformer(
        "jinaai/jina-embeddings-v3",
        trust_remote_code=True,
        model_kwargs=modelKwargs,
        cache_folder=modelCachePath
    )
model.save_pretrained(saveModelTo,safe_serialization=True)

