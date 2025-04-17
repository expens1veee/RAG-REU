from fastapi import FastAPI
from src.internal.http_server.server import Server
import uvicorn
import time
import os
import httpx


def wait_for_qdrant():
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = os.getenv("QDRANT_PORT", "6333")
    max_attempts = 30
    attempt = 0

    while attempt < max_attempts:
        try:
            with httpx.Client() as client:
                response = client.get(f"http://{qdrant_host}:{qdrant_port}/")
                if response.status_code == 200:
                    print("Qdrant is ready!")
                    return
        except httpx.RequestError:
            attempt += 1
            print(f"Waiting for Qdrant... Attempt {attempt}/{max_attempts}")
            time.sleep(2)
    raise Exception("Qdrant is not available after multiple attempts")


def main() -> None:
    wait_for_qdrant() # Ждем квадрант

    # Создаём приложение FastAPI
    app = FastAPI()

    # Создаём экземпляр класса Server и подключаем его роутер
    server = Server()
    app.include_router(server.router)

    # Запускаем сервер
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
