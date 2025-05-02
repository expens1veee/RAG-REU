from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue, ScoredPoint
from typing import Any, List
from numpy import ndarray
from torch import Tensor
import time
from src.interfaces.interfaces import IStorage
from qdrant_client.http.exceptions import UnexpectedResponse


class QdrantStorage(IStorage):
    def __init__(self, config):
        self.client = QdrantClient(url=config.qdrant_url)
        self.collection_name = config.collection_name
        self.vector_size = config.vector_size
        self._init_collection()

    def _wait_for_qdrant(self):
        for i in range(10):
            try:
                self.client.get_collections()
                print("Qdrant is up and running.")
                return
            except Exception as e:
                print(f"Waiting for Qdrant... ({i+1}/10)")
                time.sleep(1)
        raise RuntimeError("Qdrant did not become available in time.")

    def _init_collection(self):
        collections = self.client.get_collections().collections
        existing_names = [col.name for col in collections]
        print(f"Existing collections: {existing_names}")
        if self.collection_name not in existing_names:
            print(f"Creating collection '{self.collection_name}'")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )
        else:
            print(f"Collection '{self.collection_name}' already exists.")

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

    def save_data(self, embeddings: Tensor | ndarray | List[Tensor], chunks: List[str], metadata: dict):
        points = []
        for i, (embedding, chunk, meta) in enumerate(zip(embeddings, chunks, metadata)):
            payload = meta.copy()  # метаданные — это dict
            payload['text'] = chunk  # добавляем текст к метаданным
            point = PointStruct(
                id=i,
                vector=embedding,
                payload=payload,
            )
            points.append(point)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
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
