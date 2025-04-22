from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from src.internal.storage.qdrant import QdrantStorage
import numpy as np


class AskRequest(BaseModel):
    query: str
    token: Optional[str] = None  # Опциональное поле для токена


class Server:
    def __init__(self, storage: QdrantStorage):
        self.storage = storage
        # Создаём роутер с префиксом /api
        self.router = APIRouter(
            prefix="/api",
            tags=["api"],
        )
        # Регистрируем маршруты
        self._register_routes()

    def _register_routes(self):
        # Эндпоинт для проверки состояния сервера
        @self.router.get("/status")
        async def status():
            return {"message": "Alive"}

        # Эндпоинт для обработки запросов от пользователя
        @self.router.post("/ask")
        async def ask(request: AskRequest):
            return {"received_query": request.query, "received_token": request.token}

        @self.router.post("/debug/save-test")
        async def save_test():
            test_chunks = [
                "Что такое FastAPI и почему его выбирают?",
                "Как работает Retrieval-Augmented Generation?",
                "Основы архитектуры Qdrant и его интеграция с Python",
            ]

            fake_embeddings = np.random.rand(len(test_chunks), self.storage.vector_size).astype(np.float32)
            metadata = {"page_id": 123, "source": "debug_test"}

            self.storage.save_data(
                embeddings=fake_embeddings,
                chunks=test_chunks,
                metadata=metadata
            )
            return {"message": f"Сохранено {len(test_chunks)} чанков в Qdrant"}
