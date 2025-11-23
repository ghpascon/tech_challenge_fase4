import importlib
import logging
import os
import sys
from pathlib import Path


def get_path(relative_path: str) -> Path:
    """
    Retorna o caminho absoluto do arquivo ou diretório, levando em consideração se o
    aplicativo está sendo executado como um script normal ou como um executável.

    :param relative_path: Caminho relativo do arquivo ou diretório.
    :return: Caminho absoluto correto.
    """
    if getattr(sys, "frozen", False):
        # Quando o aplicativo é executado como executável (PyInstaller)
        base_path = Path(sys._MEIPASS)  # O diretório temporário onde o executável é descompactado
    else:
        # Quando o aplicativo está sendo executado do código-fonte
        base_path = Path(sys.argv[0]).resolve().parent  # O diretório onde o script foi executado

    return base_path / relative_path


def get_prefix_from_path(current_file: str, base_dir: str = "routers") -> str:
    """
    Gera automaticamente o prefixo para o APIRouter com base na estrutura de pastas a partir de 'routers'.

    :param current_file: Geralmente use __file__
    :param base_dir: Nome da pasta raiz dos routers (ex: "routers")
    :return: Prefixo do router, ex: "/rfid/get"
    """
    path = Path(current_file).resolve()
    parts = path.parts

    if base_dir not in parts:
        raise ValueError(f"'{base_dir}' not found in path: {path}")

    # Pega os subcaminhos após 'routers'
    base_index = parts.index(base_dir)
    prefix_parts = parts[base_index + 1 :]  # Ignora a pasta base e o nome do arquivo
    prefix_string = "/" + "/".join(prefix_parts)
    prefix_string = prefix_string.replace(".py", "")
    return prefix_string


# Include all routers dynamically
def include_all_routers(current_path, app):
    """
    Recursively include all routers from the given directory into the app.
    """
    routes_path = get_path(current_path)
    for entry in Path(routes_path).iterdir():
        if entry.is_dir() and entry.name != "__pycache__":
            include_all_routers(str(Path(current_path) / entry.name), app)
        elif entry.is_file() and entry.suffix == ".py" and entry.name != "__init__.py":
            module_name = entry.stem
            file_path = entry

            spec = importlib.util.spec_from_file_location(
                f"app.routers.{module_name}", str(file_path)
            )
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
                if hasattr(module, "router"):
                    prefix = getattr(module.router, "prefix", "") or ""
                    app.include_router(module.router, include_in_schema=prefix.startswith("/api"))
                    # Show path relative to 'routers' directory
                    try:
                        routers_dir = Path(routes_path).resolve()
                        relative_path = file_path.resolve().relative_to(routers_dir.parent)
                    except Exception:
                        relative_path = file_path.name
                    logging.info(f"✅ Route loaded: {relative_path}")
                else:
                    logging.warning(f"⚠️  File {relative_path} does not contain a 'router'")
            except Exception as e:
                logging.error(f"❌ Error loading {current_path}: {e}")


def load_swagger_description(swagger_file_path: str) -> str:
    """
    Loads the Swagger markdown description from file.

    Returns:
        The markdown content as a string, or a default message if file not found
    """
    try:
        with open(swagger_file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logging.warning(f"{swagger_file_path} not found. Using default description.")
        return "API documentation not found."
    except Exception as e:
        logging.error(f"Error loading Swagger documentation: {e}", exc_info=True)
        return "Error loading API documentation."
