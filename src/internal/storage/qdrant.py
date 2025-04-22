from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue, ScoredPoint
from typing import Any, List
from numpy import ndarray
from torch import Tensor

from src.interfaces.interfaces import IStorage


class QdrantStorage(IStorage):
    def __init__(self, config):
        self.client = QdrantClient(url=config.qdrant_url)
        self.collection_name = config.collection_name
        self.vector_size = config.vector_size
        self._init_collection()

    def _init_collection(self):
        if self.collection_name not in [col.name for col in self.client.get_collections().collections]:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )

    def get_data(self, query_embedding: Tensor | ndarray | List[Tensor], top_k: int) -> List[Any]:
        if isinstance(query_embedding, list):
            query_embedding = query_embedding[0]

        if isinstance(query_embedding, Tensor):
            query_embedding = query_embedding.detach().cpu().numpy()

        result: List[ScoredPoint] = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        return [point.payload for point in result]

    def save_data(self, embeddings: Tensor | ndarray | List[Tensor],
                  chunks: List[str], metadata: dict):
        if isinstance(embeddings, Tensor):
            embeddings = embeddings.detach().cpu().numpy()

        points = []
        for idx, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
            payload = metadata.copy()
            payload['text'] = chunk
            payload['chunk_index'] = idx

            points.append(PointStruct(
                id=metadata.get('page_id', 0) * 10000 + idx,
                vector=embedding,
                payload=payload
            ))

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def delete_by_page_id(self, page_id: int):
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="page_id",
                        match=MatchValue(value=page_id)
                    )
                ]
            )
        )
