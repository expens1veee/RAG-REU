from fastapi import FastAPI
from pydantic import BaseModel


class AskRequest(BaseModel):
    query: str
    token: str | None = None  # Опциональное поле для токена


app = FastAPI()


# Эндпоинт для проверки состояния сервера
@app.get("/status")
async def status():
    return {"message": "Alive"}


# Эндпоинт для обработки запросов от фронта
@app.post("/ask")
async def ask(request: AskRequest):
    # Здесь будет логика обработки запроса
    # Пока просто возвращаем полученный запрос для демонстрации
    return {"received_query": request.query, "received_token": request.token}


