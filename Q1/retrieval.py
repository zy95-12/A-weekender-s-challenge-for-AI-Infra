"""Full-vocabulary cosine nearest-neighbor attack against exposed activations."""
import torch


@torch.inference_mode()
def recover(hidden, embedding, batch=128):
    index = torch.nn.functional.normalize(embedding.float(), dim=-1)
    predicted = []
    for chunk in hidden.split(batch):
        query = torch.nn.functional.normalize(chunk.float(), dim=-1)
        predicted.extend((query @ index.T).argmax(-1).tolist())
    return predicted
