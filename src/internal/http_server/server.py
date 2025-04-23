from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from src.internal.storage.qdrant import QdrantStorage
from src.internal.retriever.retriever import Retriever


class AskRequest(BaseModel):
    query: str
    token: Optional[str] = None  # Опциональное поле для токена


class Server:
    def __init__(self, storage: QdrantStorage, retriever: Retriever = None):
        self.storage = storage
        self.retriever = retriever
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
            chunks = ["Кошка сидит на дереве", "Собака лает на прохожего", "Погода сегодня хорошая"]
            metadata = [{"source": "тест"}] * len(chunks)
            self.retriever.generate_embeddings(chunks, metadata)
            return {"message": "Сохранено"}

        @self.router.post("/debug/find-similar")
        async def find_similar():
            query = "Что делает кошка?"
            results = self.retriever.find_similar_context(query)
            return {"results": results}



