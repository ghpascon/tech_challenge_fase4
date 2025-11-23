"""
Aqui podemos colocar uma descrição sobre o projeto.

Para executar a aplicação:
    poetry run python main.py
"""

import logging
import os
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from app.core.config import settings
from app.core.fast_alerts import fast_alerts
from app.core.path import get_path, include_all_routers

# Application constants
DEFAULT_PORT = 5000
DEFAULT_HOST = "0.0.0.0"
SESSION_MAX_AGE = 3600  # 1 hour in seconds
SWAGGER_FILE_PATH = "SWAGGER.md"


def load_swagger_description() -> str:
    """
    Carrega a descrição Swagger em markdown do arquivo.

    Returns:
        O conteúdo markdown como string, ou uma mensagem padrão se o arquivo não for encontrado
    """
    try:
        with open(SWAGGER_FILE_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logging.warning(f"{SWAGGER_FILE_PATH} não encontrado. Usando descrição padrão.")
        return "Documentação não encontrada."
    except Exception as e:
        logging.error(f"Erro ao carregar documentação Swagger: {e}")
        return "Erro ao carregar documentação da API."


def create_application() -> FastAPI:
    """
    Cria e configura a aplicação FastAPI.

    Esta função:
    1. Carrega a documentação Swagger
    2. Cria a instância FastAPI
    3. Configura middleware
    4. Configura handlers de exceção
    5. Monta arquivos estáticos
    6. Inclui todos os routers

    Returns:
        Instância da aplicação FastAPI configurada
    """
    # Carrega documentação da API
    markdown_description = load_swagger_description()

    # Cria instância FastAPI
    app = FastAPI(
        title=settings.data.get("TITLE", "SMARTX"),
        description=markdown_description,
        redoc_url=None,
        docs_url=None,
    )

    # Configura middleware de sessão
    secret_key = settings.data.get("SECRET_KEY")
    if not secret_key:
        logging.warning("SECRET_KEY não encontrada nas configurações. Usando valor padrão.")
        secret_key = "default_secret_key"  # Fallback padrão

    app.add_middleware(
        SessionMiddleware,
        secret_key=secret_key,
        session_cookie="session",
        https_only=True,  # Recomendado para produção
        same_site="lax",  # Proteção básica contra CSRF
        max_age=SESSION_MAX_AGE,
    )

    # Registra handlers de exceção
    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc: Any) -> RedirectResponse:
        """Trata erros 404 Not Found redirecionando para a página inicial."""
        logging.warning(f"404 Not Found: {request.url}")
        fast_alerts.add_error(f"Rota Inválida: {request.url}")
        return RedirectResponse(url=request.app.url_path_for("index"))

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """
        Trata erros de validação de requisição com log detalhado e resposta.

        Args:
            request: A requisição recebida que falhou na validação
            exc: A exceção de validação com detalhes do erro

        Returns:
            JSONResponse com detalhes do erro de validação
        """
        # Obtém o corpo da requisição para logging
        body = await request.body()
        body_text = body.decode("utf-8", errors="ignore")

        # Registra o erro de validação com detalhes
        logging.error(
            f"Erro de validação de requisição: {request.method} {request.url}\n"
            f"Headers: {dict(request.headers)}\n"
            f"Body: {body_text}\n"
            f"Erros: {exc.errors()}"
        )

        # Retorna resposta de erro estruturada
        return JSONResponse(
            status_code=422,
            content={
                "detail": exc.errors(),
                "body": body_text,
                "message": "Requisição inválida recebida",
            },
        )

    # Monta diretório de arquivos estáticos
    static_dir = get_path("app/static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
    else:
        logging.warning(f"Diretório de arquivos estáticos não encontrado: {static_dir}")

    # Inclui todos os routers do diretório de routers
    include_all_routers("app/routers", app)
    logging.info("Aplicação configurada com sucesso")

    return app


# Cria a instância da aplicação FastAPI
app = create_application()

# Código de inicialização do servidor
if __name__ == "__main__":
    import uvicorn

    # Obtém a porta das configurações ou usa o padrão
    port = settings.data.get("PORT", DEFAULT_PORT)
    host = settings.data.get("HOST", DEFAULT_HOST)

    logging.info(f"Iniciando servidor em {host}:{port}")

    # Inicia o servidor uvicorn
    uvicorn.run(app, host=host, port=port, access_log=False)
