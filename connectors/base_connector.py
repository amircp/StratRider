from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime

class IDataConnector(ABC):
    """
    Interfaz abstracta para conectores de datos.
    Define los métodos que todos los conectores deben implementar.
    """
    
    @abstractmethod
    def get_historical_data(self, 
                           symbol: str, 
                           timeframe: str, 
                           start_date: datetime, 
                           end_date: datetime) -> pd.DataFrame:
        """
        Obtiene datos históricos para un símbolo y timeframe específicos.
        
        Args:
            symbol: Símbolo del activo (ej. 'BTC/USDT')
            timeframe: Intervalo de tiempo (ej. '1h', '1d')
            start_date: Fecha de inicio
            end_date: Fecha de fin
            
        Returns:
            DataFrame con datos OHLCV (Open, High, Low, Close, Volume)
        """
        pass
    
    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """
        Obtiene la lista de símbolos disponibles.
        
        Returns:
            Lista de símbolos disponibles
        """
        pass
    
    @abstractmethod
    def get_available_timeframes(self) -> List[str]:
        """
        Obtiene la lista de timeframes disponibles.
        
        Returns:
            Lista de timeframes disponibles
        """
        pass
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Valida si un símbolo está disponible.
        
        Args:
            symbol: Símbolo a validar
            
        Returns:
            True si el símbolo está disponible, False en caso contrario
        """
        return symbol in self.get_available_symbols()
    
    def validate_timeframe(self, timeframe: str) -> bool:
        """
        Valida si un timeframe está disponible.
        
        Args:
            timeframe: Timeframe a validar
            
        Returns:
            True si el timeframe está disponible, False en caso contrario
        """
        return timeframe in self.get_available_timeframes()