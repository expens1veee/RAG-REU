from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional


class AskRequest(BaseModel):
    query: str
    token: Optional[str] = None  # Опциональное поле для токена


class Server:
    def __init__(self):
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
