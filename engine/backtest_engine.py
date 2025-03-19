import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import logging

from strategies.base_strategy import IStrategy, Signal, SignalType

class Trade:
    """Clase para representar una operación de trading"""
    
    def __init__(self,
                entry_time: pd.Timestamp,
                entry_price: float,
                entry_type: SignalType,
                position_size: float,
                exit_time: Optional[pd.Timestamp] = None,
                exit_price: Optional[float] = None,
                exit_type: Optional[SignalType] = None,
                profit_loss: Optional[float] = None,
                profit_loss_pct: Optional[float] = None,
                metadata: Dict[str, Any] = None):
        """
        Inicializa una operación de trading.
        
        Args:
            entry_time: Tiempo de entrada
            entry_price: Precio de entrada
            entry_type: Tipo de entrada (BUY o SELL)
            position_size: Tamaño de la posición en unidades base
            exit_time: Tiempo de salida (opcional)
            exit_price: Precio de salida (opcional)
            exit_type: Tipo de salida (opcional)
            profit_loss: Beneficio/pérdida en moneda base (opcional)
            profit_loss_pct: Beneficio/pérdida en porcentaje (opcional)
            metadata: Información adicional sobre la operación
        """
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.entry_type = entry_type
        self.position_size = position_size
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_type = exit_type
        self.profit_loss = profit_loss
        self.profit_loss_pct = profit_loss_pct
        self.metadata = metadata or {}
        self.duration = None
        
        # Calcular duración si la operación está cerrada
        if exit_time is not None:
            self.duration = exit_time - entry_time
    
    def close(self, 
             exit_time: pd.Timestamp, 
             exit_price: float, 
             exit_type: SignalType) -> None:
        """
        Cierra la operación.
        
        Args:
            exit_time: Tiempo de salida
            exit_price: Precio de salida
            exit_type: Tipo de salida
        """
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_type = exit_type
        self.duration = exit_time - self.entry_time
        
        # Calcular beneficio/pérdida
        if self.entry_type == SignalType.BUY:
            # Long position
            self.profit_loss = (exit_price - self.entry_price) * self.position_size
            self.profit_loss_pct = (exit_price / self.entry_price - 1) * 100
        else:
            # Short position
            self.profit_loss = (self.entry_price - exit_price) * self.position_size
            self.profit_loss_pct = (self.entry_price / exit_price - 1) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convierte la operación a un diccionario.
        
        Returns:
            Diccionario con datos de la operación
        """
        return {
            'entry_time': self.entry_time,
            'entry_price': self.entry_price,
            'entry_type': self.entry_type.name,
            'position_size': self.position_size,
            'exit_time': self.exit_time,
            'exit_price': self.exit_price,
            'exit_type': self.exit_type.name if self.exit_type else None,
            'profit_loss': self.profit_loss,
            'profit_loss_pct': self.profit_loss_pct,
            'duration': self.duration,
            **self.metadata
        }
    
    def __str__(self) -> str:
        direction = "LONG" if self.entry_type == SignalType.BUY else "SHORT"
        status = "CLOSED" if self.exit_time else "OPEN"
        result = f"P/L: {self.profit_loss:.2f} ({self.profit_loss_pct:.2f}%)" if self.profit_loss else "P/L: N/A"
        
        return f"Trade({direction}, {status}, Entry: {self.entry_price}, Exit: {self.exit_price}, {result})"


class BacktestEngine:
    """Motor de backtesting para evaluar estrategias"""
    
    def __init__(self,
                initial_capital: float = 10000.0,
                commission_rate: float = 0.001,  # 0.1% por defecto
                slippage: float = 0.0,
                position_sizing: str = 'fixed',  # 'fixed', 'percent', 'risk_pct'
                position_size: float = 1.0,      # 100% del capital por defecto
                max_open_positions: int = 1):
        """
        Inicializa el motor de backtesting.
        
        Args:
            initial_capital: Capital inicial
            commission_rate: Tasa de comisión (por defecto 0.1%)
            slippage: Deslizamiento (por defecto 0.0)
            position_sizing: Método de dimensionamiento de posiciones
            position_size: Tamaño de la posición (significado depende de position_sizing)
            max_open_positions: Número máximo de posiciones abiertas
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.position_sizing = position_sizing
        self.position_size = position_size
        self.max_open_positions = max_open_positions
        
        # Variables de seguimiento
        self.equity = []
        self.trades = []
        self.current_position = 0  # 0: sin posición, 1: long, -1: short
        self.open_trade = None
        self.cash = initial_capital
        self.current_equity = initial_capital
    
    def run_backtest(self, data: pd.DataFrame, strategy: IStrategy) -> Dict[str, Any]:
        """
        Ejecuta un backtest con los datos y la estrategia proporcionados.
        
        Args:
            data: DataFrame con datos históricos OHLCV
            strategy: Estrategia a evaluar
            
        Returns:
            Diccionario con los resultados del backtest
        """
        # Inicializar la estrategia
        strategy.initialize(data)
        
        # Inicializar variables de seguimiento
        self.equity = []
        self.trades = []
        self.current_position = 0
        self.open_trade = None
        self.cash = self.initial_capital
        self.current_equity = self.initial_capital
        
        # Inicializar DataFrame de resultados
        result_data = data.copy()
        result_data['equity'] = np.nan
        result_data['returns'] = np.nan
        result_data['position'] = 0
        
        # Recorrer cada vela
        for idx, candle in data.iterrows():
            # Obtener la vela procesada del DataFrame de la estrategia
            if hasattr(strategy, 'data') and strategy.data is not None and idx in strategy.data.index:
                processed_candle = strategy.data.loc[idx]
            else:
                processed_candle = candle
                
            # Procesar la vela con la estrategia
            signal = strategy.process_candle(processed_candle)
            
            # Manejar señal si existe
            if signal:
                self._handle_signal(signal, candle)
            
            # Actualizar el valor de la posición actual
            current_price = candle['close']
            position_value = 0
            
            if self.open_trade:
                if self.open_trade.entry_type == SignalType.BUY:
                    # Long position
                    price_diff = current_price - self.open_trade.entry_price
                    position_value = self.open_trade.position_size * current_price
                else:
                    # Short position
                    price_diff = self.open_trade.entry_price - current_price
                    position_value = self.open_trade.position_size * self.open_trade.entry_price
            
            # Actualizar equidad
            self.current_equity = self.cash + position_value
            self.equity.append((idx, self.current_equity))
            
            # Actualizar DataFrame de resultados
            result_data.loc[idx, 'equity'] = self.current_equity
            result_data.loc[idx, 'position'] = self.current_position
            
            # Calcular retornos
            if idx > data.index[0]:
                previous_equity = result_data.loc[result_data.index[result_data.index.get_loc(idx) - 1], 'equity']
                if previous_equity:
                    result_data.loc[idx, 'returns'] = self.current_equity / previous_equity - 1
        
        # Resto del código sin cambios...
        # Cerrar posición abierta al final del período
        if self.open_trade:
            last_candle = data.iloc[-1]
            self._close_position(last_candle.name, last_candle['close'], SignalType.NEUTRAL)
        
        # Convertir trades a DataFrame
        trades_df = pd.DataFrame([trade.to_dict() for trade in self.trades])
        
        # Calcular estadísticas adicionales
        equity_curve = pd.DataFrame(self.equity, columns=['timestamp', 'equity']).set_index('timestamp')
        
        return {
            'equity_curve': equity_curve,
            'trades': trades_df,
            'result_data': result_data,
            'initial_capital': self.initial_capital,
            'final_equity': self.current_equity,
            'net_profit': self.current_equity - self.initial_capital,
            'net_profit_pct': (self.current_equity / self.initial_capital - 1) * 100,
            'strategy_name': strategy.name,
            'strategy_params': strategy.get_parameters()
        }
    
    def _handle_signal(self, signal: Signal, candle: pd.Series) -> None:
        """
        Maneja una señal generada por la estrategia.
        
        Args:
            signal: Señal generada
            candle: Datos de la vela actual
        """
        timestamp = signal.timestamp
        signal_type = signal.signal_type
        price = signal.price
        
        # Aplicar slippage
        adjusted_price = self._apply_slippage(price, signal_type)
        
        # Manejar señales según el tipo y la posición actual
        if signal_type == SignalType.BUY and self.current_position <= 0:
            # Cerrar posición corta si existe
            if self.current_position < 0:
                self._close_position(timestamp, adjusted_price, signal_type)
            
            # Abrir posición larga
            self._open_position(timestamp, adjusted_price, signal_type)
            
        elif signal_type == SignalType.SELL and self.current_position >= 0:
            # Cerrar posición larga si existe
            if self.current_position > 0:
                self._close_position(timestamp, adjusted_price, signal_type)
            
            # Abrir posición corta
            self._open_position(timestamp, adjusted_price, signal_type)
    
    def _open_position(self, timestamp: pd.Timestamp, price: float, signal_type: SignalType) -> None:
        """
        Abre una nueva posición.
        
        Args:
            timestamp: Tiempo de entrada
            price: Precio de entrada
            signal_type: Tipo de entrada
        """
        # Calcular tamaño de la posición
        position_size = self._calculate_position_size(price, signal_type)
        
        # Calcular costo total incluyendo comisiones
        position_cost = position_size * price
        commission = position_cost * self.commission_rate
        total_cost = position_cost + commission
        
        # Verificar si hay suficiente capital
        if total_cost > self.cash:
            logging.warning(f"Capital insuficiente para abrir posición: {total_cost} > {self.cash}")
            position_size = (self.cash / (1 + self.commission_rate)) / price
            position_cost = position_size * price
            commission = position_cost * self.commission_rate
            total_cost = position_cost + commission
        
        # Actualizar cash
        self.cash -= total_cost
        
        # Crear trade
        self.open_trade = Trade(
            entry_time=timestamp,
            entry_price=price,
            entry_type=signal_type,
            position_size=position_size,
            metadata={
                'commission': commission
            }
        )
        
        # Actualizar posición actual
        self.current_position = 1 if signal_type == SignalType.BUY else -1
    
    def _close_position(self, timestamp: pd.Timestamp, price: float, signal_type: SignalType) -> None:
        """
        Cierra una posición abierta.
        
        Args:
            timestamp: Tiempo de salida
            price: Precio de salida
            signal_type: Tipo de salida
        """
        if not self.open_trade:
            return
        
        # Cerrar trade
        self.open_trade.close(timestamp, price, signal_type)
        
        # Calcular valor de la posición y comisión
        position_value = self.open_trade.position_size * price
        commission = position_value * self.commission_rate
        
        # Actualizar cash
        self.cash += position_value - commission
        
        # Añadir comisión de cierre a los metadatos
        self.open_trade.metadata['exit_commission'] = commission
        self.open_trade.metadata['total_commission'] = self.open_trade.metadata.get('commission', 0) + commission
        
        # Guardar trade cerrado
        self.trades.append(self.open_trade)
        
        # Resetear posición
        self.open_trade = None
        self.current_position = 0
    
    def _calculate_position_size(self, price: float, signal_type: SignalType) -> float:
        """
        Calcula el tamaño de la posición según el método seleccionado.
        
        Args:
            price: Precio de entrada
            signal_type: Tipo de entrada
            
        Returns:
            Tamaño de la posición en unidades base
        """
        if self.position_sizing == 'fixed':
            # Tamaño fijo en unidades
            return self.position_size
        
        elif self.position_sizing == 'percent':
            # Porcentaje del capital
            capital_to_use = self.cash * self.position_size
            return capital_to_use / price / (1 + self.commission_rate)
        
        elif self.position_sizing == 'risk_pct':
            # Porcentaje de riesgo (No implementado completamente)
            # Requiere stop loss para calcular riesgo
            capital_to_risk = self.cash * self.position_size
            # Usar un valor de riesgo por defecto (1% del precio)
            risk_per_unit = price * 0.01
            return capital_to_risk / risk_per_unit
        
        else:
            # Por defecto, usar todo el capital disponible
            return self.cash / price / (1 + self.commission_rate)
    
    def _apply_slippage(self, price: float, signal_type: SignalType) -> float:
        """
        Aplica deslizamiento al precio.
        
        Args:
            price: Precio original
            signal_type: Tipo de señal
            
        Returns:
            Precio ajustado con deslizamiento
        """
        if signal_type == SignalType.BUY:
            # Para compras, el precio es más alto
            return price * (1 + self.slippage)
        else:
            # Para ventas, el precio es más bajo
            return price * (1 - self.slippage)