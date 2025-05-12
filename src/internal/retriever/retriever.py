from sentence_transformers import SentenceTransformer, util
from torch import Tensor
from typing import List, Tuple
from src.interfaces.interfaces import IRetriever, IStorage


class Retriever(IRetriever):
    def __init__(self, storage: IStorage, model_name: str = "sergeyzh/rubert-mini-frida"):
        self.model = SentenceTransformer(model_name)
        self.storage = storage

    def generate_embeddings(self, chunks: List[str], metadata: List[dict]) -> Tensor:
        if not chunks:
            return Tensor()
        embeddings = self.model.encode(
            chunks,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        self.storage.save_data(embeddings, chunks, metadata)
        return embeddings

    def find_similar_context(self, query: str) -> List[Tuple[str, str]]:
        query_embedding = self.model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
        results = self.storage.get_data(query_embedding, top_k=5)

        # results — список чанков (или словарей) из стораджа
        context_pairs = []
        for r in results:
            text = r.get("text", "")
            source = r.get("metadata", {}).get("source", "unknown")
            context_pairs.append((text, source))

        return context_pairs

    def best_match(self, query_list: List[str], context_list: List[Tuple[str, str]], top_k: int) -> str:
        """
        Простейшая реализация: берём контексты, ранжируем по косинусной близости и возвращаем топ-1.
        """
        if not context_list:
            return ""
        context_texts = [ctx[0] for ctx in context_list]
        context_embeddings = self.model.encode(context_texts, convert_to_tensor=True, normalize_embeddings=True)
        query_embeddings = self.model.encode(query_list, convert_to_tensor=True, normalize_embeddings=True)

        scores = util.cos_sim(query_embeddings, context_embeddings)
        best_score_idx = scores.argmax().item()
        return context_texts[best_score_idx]
