from fastapi import FastAPI
from src.internal.http_server.server import Server
from src.internal.storage.qdrant import QdrantStorage
import uvicorn
from dataclasses import dataclass
import os
from src.storage_config.config import StorageConfig


def main() -> None:
    config = StorageConfig(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333))
    )

    storage = QdrantStorage(config)

    # Создаём приложение FastAPI
    app = FastAPI()

    # Создаём экземпляр класса Server и подключаем его роутер
    server = Server(storage=storage)
    app.include_router(server.router)

    # Запускаем сервер
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
