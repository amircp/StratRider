import pandas as pd
import numpy as np
import ccxt
import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
import logging

from connectors.base_connector import IDataConnector

class CryptoConnector(IDataConnector):
    """
    Conector para obtener datos históricos de criptomonedas a través de la biblioteca ccxt.
    Implementa métodos para superar límites de API obteniendo datos en fragmentos.
    """
    
    def __init__(self, exchange_id: str = 'binance', api_key: str = None, api_secret: str = None):
        """
        Inicializa el conector para un exchange específico.
        
        Args:
            exchange_id: ID del exchange (por defecto 'binance')
            api_key: Clave API opcional
            api_secret: Secreto API opcional
        """
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.api_secret = api_secret
        
        # Inicializar el exchange
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}  # Se puede cambiar a 'future' para futuros
        })
        
        # Cargar mercados (necesario para algunas operaciones)
        try:
            self.exchange.load_markets()
            logging.info(f"Conectado exitosamente a {exchange_id}")
        except Exception as e:
            logging.error(f"Error al conectar con {exchange_id}: {e}")
            raise
    
    def get_historical_data(self, 
                           symbol: str, 
                           timeframe: str, 
                           start_date: datetime, 
                           end_date: datetime) -> pd.DataFrame:
        """
        Obtiene datos históricos para un símbolo y timeframe específicos.
        Implementa fragmentación para superar límites de API.
        
        Args:
            symbol: Símbolo del activo (ej. 'BTC/USDT')
            timeframe: Intervalo de tiempo (ej. '1h', '1d')
            start_date: Fecha de inicio
            end_date: Fecha de fin
            
        Returns:
            DataFrame con datos OHLCV (Open, High, Low, Close, Volume)
        """
        # Validar símbolo y timeframe
        if not self.validate_symbol(symbol):
            raise ValueError(f"Símbolo no válido: {symbol}")
        if not self.validate_timeframe(timeframe):
            raise ValueError(f"Timeframe no válido: {timeframe}")
        
        # Usar método de fragmentación para superar límites de API
        return self.get_chunked_historical_data(symbol, timeframe, start_date, end_date)
    
    def get_chunked_historical_data(self, 
                                   symbol: str, 
                                   timeframe: str, 
                                   start_date: datetime, 
                                   end_date: datetime,
                                   chunk_size: int = 1000) -> pd.DataFrame:
        """
        Obtiene datos históricos en fragmentos para superar límites de API.
        
        Args:
            symbol: Símbolo del activo
            timeframe: Intervalo de tiempo
            start_date: Fecha de inicio
            end_date: Fecha de fin
            chunk_size: Tamaño de cada fragmento (por defecto 1000)
            
        Returns:
            DataFrame combinado con todos los datos
        """
        # Convertir fechas a timestamps en milisegundos
        start_ts = int(start_date.timestamp() * 1000)
        end_ts = int(end_date.timestamp() * 1000)
        
        # Calcular el incremento de tiempo basado en el timeframe
        tf_seconds = self._timeframe_to_seconds(timeframe)
        increment_ms = tf_seconds * 1000 * chunk_size
        
        all_candles = []
        current_start_ts = start_ts
        
        while current_start_ts < end_ts:
            # Calcular el final del fragmento actual
            current_end_ts = min(current_start_ts + increment_ms, end_ts)
            
            try:
                # Obtener datos para el fragmento actual
                candles = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_start_ts,
                    limit=chunk_size
                )
                
                if not candles:
                    break
                
                all_candles.extend(candles)
                
                # Actualizar el timestamp de inicio para el próximo fragmento
                last_candle_ts = candles[-1][0]
                current_start_ts = last_candle_ts + tf_seconds * 1000
                
                # Evitar hacer demasiadas solicitudes por segundo
                time.sleep(self.exchange.rateLimit / 1000)
                
            except Exception as e:
                logging.error(f"Error al obtener datos para {symbol}: {e}")
                # Esperar un poco más en caso de error y reintentar
                time.sleep(2)
                # Si persiste el error, devolver lo que tenemos hasta ahora
                if not all_candles:
                    raise
                break
        
        # Convertir a DataFrame
        if not all_candles:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Eliminar duplicados y ordenar
        df = df[~df.index.duplicated(keep='first')]
        df.sort_index(inplace=True)
        
        # Filtrar por el rango de fechas solicitado
        df = df.loc[start_date:end_date]
        
        return df
    
    def get_available_symbols(self) -> List[str]:
        """
        Obtiene la lista de símbolos disponibles en el exchange.
        
        Returns:
            Lista de símbolos disponibles
        """
        try:
            markets = self.exchange.load_markets()
            return list(markets.keys())
        except Exception as e:
            logging.error(f"Error al obtener símbolos disponibles: {e}")
            return []
    
    def get_available_timeframes(self) -> List[str]:
        """
        Obtiene la lista de timeframes disponibles en el exchange.
        
        Returns:
            Lista de timeframes disponibles
        """
        return list(self.exchange.timeframes.keys())
    
    def _timeframe_to_seconds(self, timeframe: str) -> int:
        """
        Convierte un timeframe a segundos.
        
        Args:
            timeframe: Timeframe (ej. '1h', '1d')
            
        Returns:
            Segundos equivalentes
        """
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        
        if unit == 'm':
            return value * 60
        elif unit == 'h':
            return value * 60 * 60
        elif unit == 'd':
            return value * 24 * 60 * 60
        elif unit == 'w':
            return value * 7 * 24 * 60 * 60
        else:
            raise ValueError(f"Unidad de tiempo no reconocida: {unit}")