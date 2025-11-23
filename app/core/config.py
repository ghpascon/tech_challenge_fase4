from dotenv import load_dotenv
import os


class Settings:
    def __init__(self):
        """Application settings loader and manager."""
        load_dotenv()
        self.data = {key: value for key, value in os.environ.items()}
        
        # Configurações específicas da aplicação PETR4.SA
        if "TITLE" not in self.data:
            self.data["TITLE"] = "PETR4 Predictor - Sistema LSTM"


settings = Settings()
