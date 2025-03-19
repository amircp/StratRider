import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import logging
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS

from connectors.base_connector import IDataConnector

class InfluxDBConnector(IDataConnector):
    """
    Conector para obtener datos históricos desde InfluxDB.
    Implementa la interfaz IDataConnector.
    """
    
    def __init__(self, 
                url: str, 
                token: str, 
                org: str, 
                bucket: str,
                measurement: str = 'candles',
                symbol_tag: str = 'symbol',
                timeframe_tag: str = 'timeframe'):
        """
        Inicializa el conector para InfluxDB.
        
        Args:
            url: URL del servidor InfluxDB
            token: Token de autenticación
            org: Organización
            bucket: Bucket donde se almacenan los datos
            measurement: Nombre de la medición que contiene los datos de velas
            symbol_tag: Nombre del tag que contiene el símbolo
            timeframe_tag: Nombre del tag que contiene el timeframe
        """
        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket
        self.measurement = measurement
        self.symbol_tag = symbol_tag
        self.timeframe_tag = timeframe_tag
        
        # Inicializar cliente
        self.client = None
        try:
            self.client = InfluxDBClient(url=url, token=token, org=org)
            logging.info(f"Conectado exitosamente a InfluxDB en {url}")
        except Exception as e:
            logging.error(f"Error al conectar con InfluxDB: {e}")
            raise
    
    def get_historical_data(self, 
                           symbol: str, 
                           timeframe: str, 
                           start_date: datetime, 
                           end_date: datetime) -> pd.DataFrame:
        """
        Obtiene datos históricos desde InfluxDB para un símbolo y timeframe específicos.
        
        Args:
            symbol: Símbolo del activo (ej. 'BTC/USDT')
            timeframe: Intervalo de tiempo (ej. '1h', '1d')
            start_date: Fecha de inicio
            end_date: Fecha de fin
            
        Returns:
            DataFrame con datos OHLCV
        """
        if not self.client:
            raise ValueError("Cliente InfluxDB no inicializado")
        
        # Validar símbolo y timeframe
        if not self.validate_symbol(symbol):
            raise ValueError(f"Símbolo no válido: {symbol}")
        if not self.validate_timeframe(timeframe):
            raise ValueError(f"Timeframe no válido: {timeframe}")
        
        # Crear consulta Flux
        query = f'''
        from(bucket: "{self.bucket}")
            |> range(start: {start_date.strftime("%Y-%m-%dT%H:%M:%SZ")}, stop: {end_date.strftime("%Y-%m-%dT%H:%M:%SZ")})
            |> filter(fn: (r) => r._measurement == "{self.measurement}")
            |> filter(fn: (r) => r.{self.symbol_tag} == "{symbol}")
            |> filter(fn: (r) => r.{self.timeframe_tag} == "{timeframe}")
        '''
        
        try:
            # Ejecutar consulta
            query_api = self.client.query_api()
            result = query_api.query_data_frame(query)
            
            if result.empty:
                logging.warning(f"No se encontraron datos para {symbol} en {timeframe}")
                return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Si el resultado es una lista de DataFrames, concatenarlos
            if isinstance(result, list):
                result = pd.concat(result)
            
            # Procesar resultado
            df = self._process_influx_result(result)
            
            # Verificar que tenemos columnas OHLCV
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    logging.warning(f"Columna '{col}' no encontrada en los datos")
            
            return df
            
        except Exception as e:
            logging.error(f"Error al obtener datos desde InfluxDB: {e}")
            raise
    
    def get_available_symbols(self) -> List[str]:
        """
        Obtiene la lista de símbolos disponibles en InfluxDB.
        
        Returns:
            Lista de símbolos disponibles
        """
        if not self.client:
            raise ValueError("Cliente InfluxDB no inicializado")
        
        # Consulta para obtener símbolos únicos
        query = f'''
        import "influxdata/influxdb/schema"
        
        schema.tagValues(
            bucket: "{self.bucket}",
            tag: "{self.symbol_tag}",
            predicate: (r) => r._measurement == "{self.measurement}"
        )
        '''
        
        try:
            query_api = self.client.query_api()
            result = query_api.query_data_frame(query)
            
            if result.empty:
                return []
            
            # Extraer valores únicos
            symbols = result['_value'].unique().tolist()
            return symbols
            
        except Exception as e:
            logging.error(f"Error al obtener símbolos disponibles: {e}")
            return []
    
    def get_available_timeframes(self) -> List[str]:
        """
        Obtiene la lista de timeframes disponibles en InfluxDB.
        
        Returns:
            Lista de timeframes disponibles
        """
        if not self.client:
            raise ValueError("Cliente InfluxDB no inicializado")
        
        # Consulta para obtener timeframes únicos
        query = f'''
        import "influxdata/influxdb/schema"
        
        schema.tagValues(
            bucket: "{self.bucket}",
            tag: "{self.timeframe_tag}",
            predicate: (r) => r._measurement == "{self.measurement}"
        )
        '''
        
        try:
            query_api = self.client.query_api()
            result = query_api.query_data_frame(query)
            
            if result.empty:
                return []
            
            # Extraer valores únicos
            timeframes = result['_value'].unique().tolist()
            return timeframes
            
        except Exception as e:
            logging.error(f"Error al obtener timeframes disponibles: {e}")
            return []
    
    def write_ohlcv_data(self, 
                        data: pd.DataFrame, 
                        symbol: str, 
                        timeframe: str) -> bool:
        """
        Escribe datos OHLCV en InfluxDB.
        
        Args:
            data: DataFrame con datos OHLCV
            symbol: Símbolo del activo
            timeframe: Intervalo de tiempo
            
        Returns:
            True si la operación fue exitosa, False en caso contrario
        """
        if not self.client:
            raise ValueError("Cliente InfluxDB no inicializado")
        
        # Verificar que el DataFrame tiene las columnas necesarias
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in data.columns and (col != 'timestamp' or not isinstance(data.index, pd.DatetimeIndex)):
                raise ValueError(f"Columna '{col}' no encontrada en los datos")
        
        try:
            # Preparar datos para escritura
            write_api = self.client.write_api(write_options=SYNCHRONOUS)
            
            # Si timestamp no es columna sino índice
            df = data.copy()
            if 'timestamp' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
                df['timestamp'] = df.index
            
            # Escribir cada punto
            for i, row in df.iterrows():
                time = row['timestamp'] if 'timestamp' in row else i
                
                # Crear punto
                point = {
                    "measurement": self.measurement,
                    "tags": {
                        self.symbol_tag: symbol,
                        self.timeframe_tag: timeframe
                    },
                    "time": time,
                    "fields": {
                        "open": float(row['open']),
                        "high": float(row['high']),
                        "low": float(row['low']),
                        "close": float(row['close']),
                        "volume": float(row['volume'])
                    }
                }
                
                # Escribir punto
                write_api.write(self.bucket, self.org, point)
            
            return True
            
        except Exception as e:
            logging.error(f"Error al escribir datos en InfluxDB: {e}")
            return False
    
    def _process_influx_result(self, result: pd.DataFrame) -> pd.DataFrame:
        """
        Procesa el resultado de una consulta InfluxDB y lo convierte a formato OHLCV.
        
        Args:
            result: DataFrame con resultados de InfluxDB
            
        Returns:
            DataFrame en formato OHLCV
        """
        # Si el DataFrame está vacío, devolver DataFrame vacío
        if result.empty:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Extraer datos y pivotar
        if '_field' in result.columns and '_value' in result.columns:
            # Formato estándar de respuesta InfluxDB
            pivot_df = result.pivot_table(
                index='_time', 
                columns='_field', 
                values='_value'
            ).reset_index()
            
            # Renombrar columnas
            pivot_df = pivot_df.rename(columns={'_time': 'timestamp'})
            
        else:
            # Formato alternativo (puede variar según la estructura)
            # Aquí asumimos que ya tenemos las columnas OHLCV
            pivot_df = result
        
        # Asegurar que el índice es datetime
        if 'timestamp' in pivot_df.columns:
            pivot_df.set_index('timestamp', inplace=True)
        
        # Ordenar por índice
        pivot_df.sort_index(inplace=True)
        
        return pivot_df