import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime, timedelta
import logging

from connectors.base_connector import IDataConnector


class YahooFinanceConnector(IDataConnector):
    """
    Conector para obtener datos históricos de Yahoo Finance.
    """

    def __init__(self, proxy=None):
        """
        Inicializa el conector para Yahoo Finance.

        Args:
            proxy: Proxy para las solicitudes (opcional)
        """
        self.proxy = proxy
        self.timeframes_map = {
            '1m': '1m',
            '2m': '2m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '60m': '60m',
            '90m': '90m',
            '1h': '60m',
            '1d': '1d',
            '5d': '5d',
            '1wk': '1wk',
            '1mo': '1mo',
            '3mo': '3mo'
        }

        # Límites de Yahoo Finance para diferentes timeframes
        self.timeframe_limits = {
            '1m': 7,  # 7 días
            '2m': 60,  # 60 días
            '5m': 60,  # 60 días
            '15m': 60,  # 60 días
            '30m': 60,  # 60 días
            '60m': 730,  # 2 años
            '90m': 60,  # 60 días
            '1h': 730,  # 2 años
            '1d': 10000,  # Sin límite práctico
            '5d': 10000,  # Sin límite práctico
            '1wk': 10000,  # Sin límite práctico
            '1mo': 10000,  # Sin límite práctico
            '3mo': 10000  # Sin límite práctico
        }

        logging.info("Conector de Yahoo Finance inicializado")

    def get_historical_data(self,
                            symbol: str,
                            timeframe: str,
                            start_date: datetime,
                            end_date: datetime) -> pd.DataFrame:
        """
        Obtiene datos históricos de Yahoo Finance.

        Args:
            symbol: Símbolo del activo (ej. 'AAPL', 'BTC-USD')
            timeframe: Intervalo de tiempo (ej. '1h', '1d')
            start_date: Fecha de inicio
            end_date: Fecha de fin

        Returns:
            DataFrame con datos OHLCV
        """
        # Verificar si el timeframe es válido
        if not self.validate_timeframe(timeframe):
            raise ValueError(f"Timeframe no válido: {timeframe}")

        # Mapear timeframe al formato de Yahoo Finance
        yf_timeframe = self.timeframes_map.get(timeframe, timeframe)

        # Verificar límites de tiempo para el timeframe
        days_limit = self.timeframe_limits.get(yf_timeframe, 10000)
        requested_days = (end_date - start_date).days

        if requested_days > days_limit and yf_timeframe in ['1m', '2m', '5m', '15m', '30m', '90m']:
            logging.warning(
                f"Yahoo Finance limita los datos para {yf_timeframe} a {days_limit} días. Usando datos fragmentados.")
            return self._get_chunked_data(symbol, yf_timeframe, start_date, end_date, days_limit)

        try:
            # Obtener datos de Yahoo Finance
            ticker = yf.Ticker(symbol)
            data = ticker.history(interval=yf_timeframe, start=start_date, end=end_date, proxy=self.proxy)

            # Verificar si se obtuvieron datos
            if data.empty:
                logging.warning(f"No se encontraron datos para {symbol} con timeframe {timeframe}")
                return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

            # Renombrar columnas para adecuarse al estándar del framework
            data.columns = [col.lower() for col in data.columns]

            # Asegurar que tenemos las columnas requeridas
            if all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                # Yahoo Finance a veces incluye columnas adicionales, seleccionamos solo las necesarias
                data = data[['open', 'high', 'low', 'close', 'volume']]

                return data
            else:
                logging.error(f"Falta alguna columna OHLCV en los datos de {symbol}")
                return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

        except Exception as e:
            logging.error(f"Error al obtener datos para {symbol}: {e}")
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    def _get_chunked_data(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime,
                          days_limit: int) -> pd.DataFrame:
        """
        Obtiene datos históricos en fragmentos para superar límites de API.

        Args:
            symbol: Símbolo del activo
            timeframe: Intervalo de tiempo en formato Yahoo Finance
            start_date: Fecha de inicio
            end_date: Fecha de fin
            days_limit: Límite de días para el timeframe

        Returns:
            DataFrame combinado con todos los datos
        """
        all_data = []
        current_start = start_date

        while current_start < end_date:
            current_end = current_start + timedelta(days=days_limit)
            if current_end > end_date:
                current_end = end_date

            try:
                ticker = yf.Ticker(symbol)
                chunk = ticker.history(interval=timeframe, start=current_start, end=current_end, proxy=self.proxy)

                if not chunk.empty:
                    all_data.append(chunk)

                # Actualizar fecha de inicio para el próximo fragmento
                current_start = current_end

            except Exception as e:
                logging.error(f"Error al obtener fragmento para {symbol}: {e}")
                # Continuar con el siguiente fragmento
                current_start = current_end

        if not all_data:
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

        # Combinar todos los fragmentos
        combined_data = pd.concat(all_data)

        # Renombrar columnas y seleccionar solo OHLCV
        combined_data.columns = [col.lower() for col in combined_data.columns]
        combined_data = combined_data[['open', 'high', 'low', 'close', 'volume']]

        return combined_data

    def get_available_symbols(self) -> List[str]:
        """
        Yahoo Finance no proporciona una lista completa de símbolos disponibles.
        En su lugar, devolvemos una lista de símbolos comunes.

        Returns:
            Lista de símbolos disponibles
        """
        # Lista de símbolos comunes (índices, acciones populares, crypto)
        common_symbols = [
            # Índices
            '^GSPC',  # S&P 500
            '^DJI',  # Dow Jones
            '^IXIC',  # NASDAQ
            '^FTSE',  # FTSE 100
            '^N225',  # Nikkei 225

            # Acciones populares
            'AAPL',  # Apple
            'MSFT',  # Microsoft
            'GOOGL',  # Google
            'AMZN',  # Amazon
            'META',  # Facebook (Meta)
            'TSLA',  # Tesla
            'NVDA',  # NVIDIA

            # Criptomonedas
            'BTC-USD',  # Bitcoin
            'ETH-USD',  # Ethereum
            'XRP-USD',  # Ripple
            'LTC-USD',  # Litecoin
            'ADA-USD',  # Cardano
            'SOL-USD'  # Solana
        ]

        return common_symbols

    def get_available_timeframes(self) -> List[str]:
        """
        Obtiene la lista de timeframes disponibles en Yahoo Finance.

        Returns:
            Lista de timeframes disponibles
        """
        return list(self.timeframes_map.keys())

    def get_latest_price(self, symbol: str) -> float:
        """
        Obtiene el último precio disponible para un símbolo.

        Args:
            symbol: Símbolo del activo

        Returns:
            Último precio
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d')

            if data.empty:
                raise ValueError(f"No se encontraron datos para {symbol}")

            return data['Close'].iloc[-1]
        except Exception as e:
            logging.error(f"Error al obtener el último precio para {symbol}: {e}")
            return 0.0