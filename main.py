from fastapi import FastAPI
from src.internal.http_server.server import Server
import uvicorn


def main() -> None:
    # Создаём приложение FastAPI
    app = FastAPI()

    # Создаём экземпляр класса Server и подключаем его роутер
    server = Server()
    app.include_router(server.router)

    # Запускаем сервер
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
