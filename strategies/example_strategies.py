import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import talib

from strategies.base_strategy import IStrategy, Signal, SignalType

class MovingAverageCrossover(IStrategy):
    """
    Estrategia basada en el cruce de medias móviles.
    Genera señales de compra cuando la media corta cruza por encima de la media larga,
    y señales de venta cuando la media corta cruza por debajo de la media larga.
    """
    
    def __init__(self, 
                fast_period: int = 20, 
                slow_period: int = 50,
                name: str = "MA Crossover"):
        """
        Inicializa la estrategia con los períodos de las medias móviles.
        
        Args:
            fast_period: Período de la media móvil rápida
            slow_period: Período de la media móvil lenta
            name: Nombre de la estrategia
        """
        super().__init__(name=name)
        
        self.parameters = {
            'fast_period': fast_period,
            'slow_period': slow_period
        }
        
        self.data = None
        self.position = 0  # 0: sin posición, 1: long, -1: short
    
    def initialize(self, data: pd.DataFrame) -> None:
        """
        Inicializa la estrategia con los datos históricos y calcula indicadores.
        
        Args:
            data: DataFrame con datos históricos OHLCV
        """
        self.data = data.copy()
        
        # Calcular medias móviles
        self.data['fast_ma'] = talib.SMA(self.data['close'].values, 
                                         timeperiod=self.parameters['fast_period'])
        self.data['slow_ma'] = talib.SMA(self.data['close'].values, 
                                         timeperiod=self.parameters['slow_period'])
        
        # Calcular señal (1 cuando fast_ma > slow_ma, -1 cuando fast_ma < slow_ma)
        self.data['signal'] = np.where(self.data['fast_ma'] > self.data['slow_ma'], 1, -1)
        
        # Detectar cambios en la señal (cruces)
        self.data['signal_change'] = self.data['signal'].diff()
        
        # Resetear señales
        self.signals = []
    
    def process_candle(self, candle: pd.Series) -> Optional[Signal]:
        """
        Procesa una nueva vela y genera una señal si hay un cruce.
        
        Args:
            candle: Serie de Pandas con datos de la vela actual
            
        Returns:
            Señal generada o None si no hay señal
        """
        # Obtener valores relevantes
        timestamp = candle.name
        close_price = candle['close']
        signal_change = candle['signal_change']
        
        # Verificar si hay un cruce
        if signal_change == 2:  # Cruce alcista (de -1 a 1)
            signal = Signal(
                timestamp=timestamp,
                signal_type=SignalType.BUY,
                price=close_price,
                metadata={
                    'fast_ma': candle['fast_ma'],
                    'slow_ma': candle['slow_ma']
                }
            )
            self.signals.append(signal)
            self.position = 1
            return signal
            
        elif signal_change == -2:  # Cruce bajista (de 1 a -1)
            signal = Signal(
                timestamp=timestamp,
                signal_type=SignalType.SELL,
                price=close_price,
                metadata={
                    'fast_ma': candle['fast_ma'],
                    'slow_ma': candle['slow_ma']
                }
            )
            self.signals.append(signal)
            self.position = -1
            return signal
        
        return None
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Establece los parámetros de la estrategia y recalcula indicadores.
        
        Args:
            parameters: Diccionario con los parámetros
        """
        super().set_parameters(parameters)
        
        # Si ya hay datos, recalcular indicadores
        if self.data is not None:
            self.initialize(self.data)