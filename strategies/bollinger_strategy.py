import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import talib

from strategies.base_strategy import IStrategy, Signal, SignalType

class BollingerBreakoutStrategy(IStrategy):
    """
    Estrategia de ruptura de Bandas de Bollinger.
    Genera señales de compra cuando el precio cierra por encima de la banda superior
    y señales de venta cuando cierra por debajo de la banda inferior.
    """
    
    def __init__(self, 
                period: int = 20, 
                deviations: float = 2.0,
                exit_opposite: bool = True,
                name: str = "Bollinger Breakout"):
        """
        Inicializa la estrategia.
        
        Args:
            period: Período para las Bandas de Bollinger
            deviations: Número de desviaciones estándar para las bandas
            exit_opposite: Si salir de posiciones con señales opuestas
            name: Nombre de la estrategia
        """
        super().__init__(name=name)
        
        self.parameters = {
            'period': period,
            'deviations': deviations,
            'exit_opposite': exit_opposite
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
        
        # Calcular Bandas de Bollinger
        upper, middle, lower = talib.BBANDS(
            self.data['close'].values, 
            timeperiod=self.parameters['period'], 
            nbdevup=self.parameters['deviations'], 
            nbdevdn=self.parameters['deviations'], 
            matype=0
        )
        
        self.data['bb_upper'] = upper
        self.data['bb_middle'] = middle
        self.data['bb_lower'] = lower
        
        # Calcular %B (posición relativa en las bandas)
        # %B = (Precio - Banda Inferior) / (Banda Superior - Banda Inferior)
        self.data['bb_pct_b'] = (self.data['close'] - self.data['bb_lower']) / (self.data['bb_upper'] - self.data['bb_lower'])
        
        # Calcular cruces con las bandas
        self.data['cross_upper'] = (self.data['close'] > self.data['bb_upper']) & (self.data['close'].shift(1) <= self.data['bb_upper'].shift(1))
        self.data['cross_lower'] = (self.data['close'] < self.data['bb_lower']) & (self.data['close'].shift(1) >= self.data['bb_lower'].shift(1))
        
        # Resetear señales
        self.signals = []
        self.position = 0
    
    def process_candle(self, candle: pd.Series) -> Optional[Signal]:
        """
        Procesa una nueva vela y genera una señal si hay un cruce con las bandas.
        
        Args:
            candle: Serie de Pandas con datos de la vela actual
            
        Returns:
            Señal generada o None si no hay señal
        """
        # Obtener valores relevantes
        timestamp = candle.name
        close_price = candle['close']
        cross_upper = candle['cross_upper']
        cross_lower = candle['cross_lower']
        
        # Generar señales
        if cross_upper:
            # Si el precio cruza la banda superior, señal de compra
            
            # Si estamos en posición corta y se configura salir con señal opuesta
            if self.position < 0 and self.parameters['exit_opposite']:
                # Primero cerrar posición corta
                signal = Signal(
                    timestamp=timestamp,
                    signal_type=SignalType.BUY,  # Comprar para cerrar corto
                    price=close_price,
                    metadata={
                        'bb_upper': candle['bb_upper'],
                        'bb_lower': candle['bb_lower'],
                        'bb_pct_b': candle['bb_pct_b'],
                        'action': 'exit_short'
                    }
                )
                self.signals.append(signal)
                self.position = 0
                return signal
            
            # Si no estamos ya en posición larga
            elif self.position <= 0:
                signal = Signal(
                    timestamp=timestamp,
                    signal_type=SignalType.BUY,
                    price=close_price,
                    metadata={
                        'bb_upper': candle['bb_upper'],
                        'bb_lower': candle['bb_lower'],
                        'bb_pct_b': candle['bb_pct_b'],
                        'action': 'enter_long'
                    }
                )
                self.signals.append(signal)
                self.position = 1
                return signal
        
        elif cross_lower:
            # Si el precio cruza la banda inferior, señal de venta
            
            # Si estamos en posición larga y se configura salir con señal opuesta
            if self.position > 0 and self.parameters['exit_opposite']:
                # Primero cerrar posición larga
                signal = Signal(
                    timestamp=timestamp,
                    signal_type=SignalType.SELL,  # Vender para cerrar largo
                    price=close_price,
                    metadata={
                        'bb_upper': candle['bb_upper'],
                        'bb_lower': candle['bb_lower'],
                        'bb_pct_b': candle['bb_pct_b'],
                        'action': 'exit_long'
                    }
                )
                self.signals.append(signal)
                self.position = 0
                return signal
            
            # Si no estamos ya en posición corta
            elif self.position >= 0:
                signal = Signal(
                    timestamp=timestamp,
                    signal_type=SignalType.SELL,
                    price=close_price,
                    metadata={
                        'bb_upper': candle['bb_upper'],
                        'bb_lower': candle['bb_lower'],
                        'bb_pct_b': candle['bb_pct_b'],
                        'action': 'enter_short'
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