import os

from sentence_transformers import util

from typing import List, Tuple
from src.interfaces.interfaces import IRetriever, IStorage
import numpy as np
from yandex_cloud_ml_sdk import YCloudML

sdk = YCloudML(
        folder_id=os.getenv("folder_id"),
        auth=os.getenv("api"),
    )


class Retriever(IRetriever):
    def __init__(self, storage: IStorage):
        self.storage = storage

    def generate_embeddings(self, chunks: List[str], metadata: List[dict]):
        doc_model = sdk.models.text_embeddings("doc")
        doc_embeddings = [doc_model.run(text) for text in chunks]
        self.storage.save_data(doc_embeddings, chunks, metadata)
        return doc_embeddings

    def find_similar_context(self, query: str) -> List[Tuple[str, str]]:
        query_model = sdk.models.text_embeddings("query")
        query_embedding = np.array(query_model.run(query))
        results = self.storage.get_data(query_embedding, top_k=3)

        # results — список чанков (или словарей) из стораджа
        context_pairs = []
        for r in results:
            text = r.get("text", "")
            source = r.get("metadata", {}).get("source", "unknown")
            context_pairs.append((text, source))

        return context_pairs

    def best_match(self, query, context_list: List[Tuple[str, str]], top_k: int) -> str:
        """
        Простейшая реализация: берём контексты, ранжируем по косинусной близости и возвращаем топ-1.
        """
        if not context_list:
            return ""
        context_texts = [ctx[0] for ctx in context_list]

        doc_model = sdk.models.text_embeddings("doc")
        context_embeddings = [doc_model.run(str(text)) for text in context_list]
        query_model = sdk.models.text_embeddings("query")
        query_embedding = np.array(query_model.run(query))

        context_embeddings = np.array(context_embeddings)

        scores = util.cos_sim(query_embedding, context_embeddings)
        best_score_idx = scores.argmax().item()
        return context_texts[best_score_idx]



