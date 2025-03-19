from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from enum import Enum

class SignalType(Enum):
    """Tipos de señales que puede generar una estrategia"""
    BUY = 1
    SELL = -1
    NEUTRAL = 0

class Signal:
    """Clase para representar una señal de trading"""
    
    def __init__(self, 
                 timestamp: pd.Timestamp, 
                 signal_type: SignalType, 
                 price: float, 
                 size: float = 1.0,
                 metadata: Dict[str, Any] = None):
        """
        Inicializa una señal de trading.
        
        Args:
            timestamp: Marca de tiempo de la señal
            signal_type: Tipo de señal (BUY, SELL, NEUTRAL)
            price: Precio al que se genera la señal
            size: Tamaño relativo de la posición (por defecto 1.0)
            metadata: Información adicional sobre la señal
        """
        self.timestamp = timestamp
        self.signal_type = signal_type
        self.price = price
        self.size = size
        self.metadata = metadata or {}
    
    def __str__(self) -> str:
        return f"Signal({self.signal_type.name}, {self.timestamp}, price={self.price}, size={self.size})"

class IStrategy(ABC):
    """
    Interfaz abstracta para estrategias de trading.
    Define los métodos que todas las estrategias deben implementar.
    """
    
    def __init__(self, name: str = "BaseStrategy"):
        """
        Inicializa la estrategia.
        
        Args:
            name: Nombre de la estrategia
        """
        self.name = name
        self.signals = []
        self.parameters = {}
    
    @abstractmethod
    def initialize(self, data: pd.DataFrame) -> None:
        """
        Inicializa la estrategia con los datos históricos.
        Se llama una vez antes de comenzar el backtesting.
        
        Args:
            data: DataFrame con datos históricos
        """
        pass
    
    @abstractmethod
    def process_candle(self, candle: pd.Series) -> Optional[Signal]:
        """
        Procesa una nueva vela y genera una señal si es necesario.
        Se llama para cada vela durante el backtesting.
        
        Args:
            candle: Serie de Pandas con datos de la vela actual
            
        Returns:
            Señal generada o None si no hay señal
        """
        pass
    
    def get_signals(self) -> List[Signal]:
        """
        Obtiene todas las señales generadas por la estrategia.
        
        Returns:
            Lista de señales generadas
        """
        return self.signals
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Obtiene los parámetros actuales de la estrategia.
        
        Returns:
            Diccionario con los parámetros
        """
        return self.parameters
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Establece los parámetros de la estrategia.
        
        Args:
            parameters: Diccionario con los parámetros
        """
        self.parameters.update(parameters)