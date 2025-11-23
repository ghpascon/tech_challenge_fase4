import threading
from typing import Dict, List


class FastAlerts:
    def __init__(self):
        self.alerts: List[Dict[str, str]] = []
        self._lock = threading.Lock()
        self._valid_styles = {
            "alert-primary",
            "alert-secondary",
            "alert-success",
            "alert-danger",
            "alert-warning",
            "alert-info",
            "alert-light",
            "alert-dark",
        }

    def add_alert(self, detail: str, style: str = "alert-primary"):
        """
        Add an alert message with a specified Bootstrap style.

        Parameters:
        - detail (str): The message content to display.
        - style (str): Bootstrap alert style. One of the following:
            - 'alert-primary'   (default blue)
            - 'alert-secondary' (neutral gray)
            - 'alert-success'   (green for success)
            - 'alert-danger'    (red for errors)
            - 'alert-warning'   (yellow for warnings)
            - 'alert-info'      (light blue for information)
            - 'alert-light'     (light background)
            - 'alert-dark'      (dark background)

        Note: If an invalid style is provided, defaults to 'alert-primary'.
        """
        # Valida o style, usa primary como fallback se inválido
        if style not in self._valid_styles:
            style = "alert-primary"

        with self._lock:
            self.alerts.append({"detail": detail, "style": style})

    def get_alerts(self) -> List[Dict[str, str]]:
        """
        Return all current alerts and clear the list.

        Returns:
        - List[Dict[str, str]]: A list of dictionaries with 'detail' and 'style' keys.
        """
        with self._lock:
            alerts = self.alerts.copy()
            self.alerts.clear()
            return alerts

    def clear_alerts(self):
        """
        Clear all stored alerts.
        """
        with self._lock:
            self.alerts.clear()

    def add_success(self, detail: str):
        self.add_alert(detail=detail, style="alert-success")

    def add_error(self, detail: str):
        self.add_alert(detail=detail, style="alert-danger")

fast_alerts = FastAlerts()
