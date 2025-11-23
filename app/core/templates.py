import logging
import os
from datetime import datetime
from typing import Any, Callable, Dict

from fastapi.templating import Jinja2Templates

from .fast_alerts import fast_alerts
from .path import get_path


def generate_footer() -> str:
    """
    Generate footer text with current year.
    """
    year = datetime.now().year
    return f"© {year} - FIAP TECH CHALLENGE FASE 3"


def get_template_globals() -> Dict[str, Any]:
    """
    Get all template global functions and variables.

    Returns:
        Dict[str, Any]: Dictionary of global template functions
    """
    return {
        "generate_footer": generate_footer,
        "get_alerts": fast_alerts.get_alerts,
    }


def create_templates() -> Jinja2Templates:
    """
    Create and configure Jinja2Templates instance.

    Returns:
        Jinja2Templates: Configured templates instance

    Raises:
        FileNotFoundError: If template directory doesn't exist
        Exception: If template initialization fails
    """
    template_dir = get_path("app/templates")

    # Verify template directory exists
    if not os.path.exists(template_dir):
        error_msg = f"Template directory not found: {template_dir}"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        # Create templates instance
        templates_instance = Jinja2Templates(directory=template_dir)

        # Register all global functions
        template_globals = get_template_globals()
        for name, func in template_globals.items():
            templates_instance.env.globals[name] = func

        logging.info(f"Templates initialized successfully from: {template_dir}")
        logging.debug(
            f"Registered {len(template_globals)} template globals: {list(template_globals.keys())}"
        )

        return templates_instance

    except Exception as e:
        error_msg = f"Failed to initialize templates: {e}"
        logging.error(error_msg)
        raise Exception(error_msg) from e


def add_template_global(name: str, func: Callable) -> None:
    """
    Add a new global function to templates.

    Args:
        name: Name of the global function
        func: Function to add as global
    """
    if hasattr(templates, "env"):
        templates.env.globals[name] = func
        logging.debug(f"Added template global: {name}")
    else:
        logging.warning(f"Cannot add template global '{name}': templates not initialized")


# Initialize templates instance
templates = create_templates()
