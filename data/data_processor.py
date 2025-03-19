import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
import talib
import logging

class DataProcessor:
    """
    Clase para procesar y normalizar datos para backtesting.
    Permite añadir indicadores técnicos y transformar datos.
    """
    
    def __init__(self, verbose: bool = False):
        """
        Inicializa el procesador de datos.
        
        Args:
            verbose: Si mostrar logs detallados
        """
        self.verbose = verbose
        self.logger = logging.getLogger('DataProcessor')
        
        # Registro de indicadores disponibles
        self._available_indicators = {
            # Indicadores de tendencia
            'sma': self._add_sma,
            'ema': self._add_ema,
            'macd': self._add_macd,
            'adx': self._add_adx,
            
            # Osciladores
            'rsi': self._add_rsi,
            'stoch': self._add_stochastic,
            'cci': self._add_cci,
            'williams_r': self._add_williams_r,
            
            # Volatilidad
            'bollinger': self._add_bollinger_bands,
            'atr': self._add_atr,
            
            # Volumen
            'obv': self._add_obv,
            'ad': self._add_ad,
            
            # Patrones de velas
            'doji': self._add_doji,
            'hammer': self._add_hammer,
            'engulfing': self._add_engulfing
        }
    
    def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza un DataFrame de datos para asegurar formato consistente.
        
        Args:
            data: DataFrame con datos OHLCV
            
        Returns:
            DataFrame normalizado
        """
        # Hacer copia para no modificar el original
        df = data.copy()
        
        # Asegurar que las columnas tienen nombres estándar (minúsculas)
        df.columns = [col.lower() for col in df.columns]
        
        # Asegurar que existe columnas OHLCV
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"La columna '{col}' es requerida pero no está presente en los datos")
        
        # Asegurar que el índice es de tipo datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            # Buscar columna de timestamp
            timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
            if timestamp_cols:
                df.set_index(timestamp_cols[0], inplace=True)
                df.index = pd.to_datetime(df.index)
            else:
                raise ValueError("No se pudo encontrar una columna de timestamp para establecer como índice")
        
        # Ordenar por índice
        df.sort_index(inplace=True)
        
        # Eliminar filas duplicadas en el índice
        df = df[~df.index.duplicated(keep='first')]
        
        # Eliminar filas con valores NaN en columnas importantes
        df.dropna(subset=required_columns, inplace=True)
        
        return df
    
    def resample_data(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Remuestrea datos a un timeframe diferente.
        
        Args:
            data: DataFrame con datos OHLCV
            timeframe: Nuevo timeframe (e.g., '1H', '4H', '1D')
            
        Returns:
            DataFrame remuestreado
        """
        # Asegurar que el índice es de tipo datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("El índice del DataFrame debe ser de tipo datetime para remuestrear")
        
        # Hacer copia para no modificar el original
        df = data.copy()
        
        # Remuestrear datos
        ohlc_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Filtrar para usar solo las columnas que existen en el DataFrame
        ohlc_dict = {k: v for k, v in ohlc_dict.items() if k in df.columns}
        
        # Remuestrear
        resampled = df.resample(timeframe).agg(ohlc_dict)
        
        # Eliminar períodos sin datos
        resampled.dropna(subset=['close'], inplace=True)
        
        return resampled
    
    def add_indicators(self, data: pd.DataFrame, indicators: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Añade indicadores técnicos al DataFrame.
        
        Args:
            data: DataFrame con datos OHLCV
            indicators: Lista de diccionarios con indicadores a añadir
                        Cada diccionario debe tener al menos una clave 'type'
                        
        Returns:
            DataFrame con indicadores añadidos
        """
        # Hacer copia para no modificar el original
        df = data.copy()
        
        # Añadir cada indicador
        for indicator in indicators:
            indicator_type = indicator.get('type')
            
            if indicator_type in self._available_indicators:
                # Extraer parámetros del indicador
                params = {k: v for k, v in indicator.items() if k != 'type'}
                
                # Llamar a la función correspondiente
                df = self._available_indicators[indicator_type](df, **params)
            else:
                self.logger.warning(f"Indicador '{indicator_type}' no disponible")
        
        return df
    
    def _add_sma(self, 
                df: pd.DataFrame, 
                period: int = 20, 
                column: str = 'close', 
                name: Optional[str] = None) -> pd.DataFrame:
        """
        Añade una Media Móvil Simple (SMA).
        
        Args:
            df: DataFrame con datos
            period: Período de la media móvil
            column: Columna sobre la que calcular la media
            name: Nombre personalizado para la columna (opcional)
            
        Returns:
            DataFrame con la SMA añadida
        """
        if column not in df.columns:
            self.logger.warning(f"Columna '{column}' no encontrada para calcular SMA")
            return df
        
        # Calcular SMA
        col_name = name or f'sma_{period}'
        df[col_name] = talib.SMA(df[column].values, timeperiod=period)
        
        return df
    
    def _add_ema(self, 
                df: pd.DataFrame, 
                period: int = 20, 
                column: str = 'close', 
                name: Optional[str] = None) -> pd.DataFrame:
        """
        Añade una Media Móvil Exponencial (EMA).
        
        Args:
            df: DataFrame con datos
            period: Período de la media móvil
            column: Columna sobre la que calcular la media
            name: Nombre personalizado para la columna (opcional)
            
        Returns:
            DataFrame con la EMA añadida
        """
        if column not in df.columns:
            self.logger.warning(f"Columna '{column}' no encontrada para calcular EMA")
            return df
        
        # Calcular EMA
        col_name = name or f'ema_{period}'
        df[col_name] = talib.EMA(df[column].values, timeperiod=period)
        
        return df
    
    def _add_macd(self, 
                 df: pd.DataFrame, 
                 fast_period: int = 12, 
                 slow_period: int = 26, 
                 signal_period: int = 9,
                 column: str = 'close') -> pd.DataFrame:
        """
        Añade el indicador MACD (Moving Average Convergence Divergence).
        
        Args:
            df: DataFrame con datos
            fast_period: Período de la media rápida
            slow_period: Período de la media lenta
            signal_period: Período de la línea de señal
            column: Columna sobre la que calcular el MACD
            
        Returns:
            DataFrame con el MACD añadido
        """
        if column not in df.columns:
            self.logger.warning(f"Columna '{column}' no encontrada para calcular MACD")
            return df
        
        # Calcular MACD
        macd, signal, hist = talib.MACD(
            df[column].values, 
            fastperiod=fast_period, 
            slowperiod=slow_period, 
            signalperiod=signal_period
        )
        
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        
        return df
    
    def _add_rsi(self, 
                df: pd.DataFrame, 
                period: int = 14, 
                column: str = 'close',
                name: Optional[str] = None) -> pd.DataFrame:
        """
        Añade el indicador RSI (Relative Strength Index).
        
        Args:
            df: DataFrame con datos
            period: Período del RSI
            column: Columna sobre la que calcular el RSI
            name: Nombre personalizado para la columna (opcional)
            
        Returns:
            DataFrame con el RSI añadido
        """
        if column not in df.columns:
            self.logger.warning(f"Columna '{column}' no encontrada para calcular RSI")
            return df
        
        # Calcular RSI
        col_name = name or f'rsi_{period}'
        df[col_name] = talib.RSI(df[column].values, timeperiod=period)
        
        return df
    
    def _add_bollinger_bands(self, 
                            df: pd.DataFrame, 
                            period: int = 20, 
                            deviations: float = 2.0,
                            column: str = 'close') -> pd.DataFrame:
        """
        Añade Bandas de Bollinger.
        
        Args:
            df: DataFrame con datos
            period: Período de las bandas
            deviations: Número de desviaciones estándar
            column: Columna sobre la que calcular las bandas
            
        Returns:
            DataFrame con las Bandas de Bollinger añadidas
        """
        if column not in df.columns:
            self.logger.warning(f"Columna '{column}' no encontrada para calcular Bandas de Bollinger")
            return df
        
        # Calcular Bandas de Bollinger
        upper, middle, lower = talib.BBANDS(
            df[column].values, 
            timeperiod=period, 
            nbdevup=deviations, 
            nbdevdn=deviations, 
            matype=0
        )
        
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        
        # Añadir porcentaje B (posición relativa en las bandas)
        df['bb_pct_b'] = (df[column] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    def _add_stochastic(self, 
                       df: pd.DataFrame, 
                       k_period: int = 14, 
                       d_period: int = 3, 
                       slowing: int = 3) -> pd.DataFrame:
        """
        Añade el oscilador estocástico.
        
        Args:
            df: DataFrame con datos
            k_period: Período de %K
            d_period: Período de %D
            slowing: Período de ralentización
            
        Returns:
            DataFrame con el estocástico añadido
        """
        # Comprobar que existen las columnas necesarias
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                self.logger.warning(f"Columna '{col}' no encontrada para calcular Estocástico")
                return df
        
        # Calcular Estocástico
        slowk, slowd = talib.STOCH(
            df['high'].values, 
            df['low'].values, 
            df['close'].values, 
            fastk_period=k_period, 
            slowk_period=slowing, 
            slowk_matype=0, 
            slowd_period=d_period, 
            slowd_matype=0
        )
        
        df['stoch_k'] = slowk
        df['stoch_d'] = slowd
        
        return df
    
    def _add_adx(self, 
                df: pd.DataFrame, 
                period: int = 14) -> pd.DataFrame:
        """
        Añade el indicador ADX (Average Directional Index).
        
        Args:
            df: DataFrame con datos
            period: Período del ADX
            
        Returns:
            DataFrame con el ADX añadido
        """
        # Comprobar que existen las columnas necesarias
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                self.logger.warning(f"Columna '{col}' no encontrada para calcular ADX")
                return df
        
        # Calcular ADX
        df['adx'] = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)
        
        # Calcular +DI y -DI
        df['plus_di'] = talib.PLUS_DI(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)
        df['minus_di'] = talib.MINUS_DI(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)
        
        return df
    
    def _add_atr(self, 
                df: pd.DataFrame, 
                period: int = 14) -> pd.DataFrame:
        """
        Añade el indicador ATR (Average True Range).
        
        Args:
            df: DataFrame con datos
            period: Período del ATR
            
        Returns:
            DataFrame con el ATR añadido
        """
        # Comprobar que existen las columnas necesarias
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                self.logger.warning(f"Columna '{col}' no encontrada para calcular ATR")
                return df
        
        # Calcular ATR
        df['atr'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)
        
        return df
    
    def _add_cci(self, 
                df: pd.DataFrame, 
                period: int = 14) -> pd.DataFrame:
        """
        Añade el indicador CCI (Commodity Channel Index).
        
        Args:
            df: DataFrame con datos
            period: Período del CCI
            
        Returns:
            DataFrame con el CCI añadido
        """
        # Comprobar que existen las columnas necesarias
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                self.logger.warning(f"Columna '{col}' no encontrada para calcular CCI")
                return df
        
        # Calcular CCI
        df['cci'] = talib.CCI(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)
        
        return df
    
    def _add_williams_r(self, 
                       df: pd.DataFrame, 
                       period: int = 14) -> pd.DataFrame:
        """
        Añade el indicador Williams %R.
        
        Args:
            df: DataFrame con datos
            period: Período del Williams %R
            
        Returns:
            DataFrame con el Williams %R añadido
        """
        # Comprobar que existen las columnas necesarias
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                self.logger.warning(f"Columna '{col}' no encontrada para calcular Williams %R")
                return df
        
        # Calcular Williams %R
        df['williams_r'] = talib.WILLR(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)
        
        return df
    
    def _add_obv(self, 
                df: pd.DataFrame) -> pd.DataFrame:
        """
        Añade el indicador OBV (On-Balance Volume).
        
        Args:
            df: DataFrame con datos
            
        Returns:
            DataFrame con el OBV añadido
        """
        # Comprobar que existen las columnas necesarias
        required_cols = ['close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                self.logger.warning(f"Columna '{col}' no encontrada para calcular OBV")
                return df
        
        # Calcular OBV
        df['obv'] = talib.OBV(df['close'].values, df['volume'].values)
        
        return df
    
    def _add_ad(self, 
               df: pd.DataFrame) -> pd.DataFrame:
        """
        Añade el indicador A/D (Accumulation/Distribution Line).
        
        Args:
            df: DataFrame con datos
            
        Returns:
            DataFrame con el A/D añadido
        """
        # Comprobar que existen las columnas necesarias
        required_cols = ['high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                self.logger.warning(f"Columna '{col}' no encontrada para calcular A/D")
                return df
        
        # Calcular A/D
        df['ad'] = talib.AD(df['high'].values, df['low'].values, df['close'].values, df['volume'].values)
        
        return df
    
    def _add_doji(self, 
                 df: pd.DataFrame, 
                 threshold: float = 0.1) -> pd.DataFrame:
        """
        Añade detector de patrones Doji.
        
        Args:
            df: DataFrame con datos
            threshold: Umbral para considerar un Doji (como % del rango diario)
            
        Returns:
            DataFrame con detector de Doji añadido
        """
        # Comprobar que existen las columnas necesarias
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                self.logger.warning(f"Columna '{col}' no encontrada para detectar Doji")
                return df
        
        # Calcular tamaño del cuerpo y del rango
        df['body_size'] = abs(df['close'] - df['open'])
        df['range_size'] = df['high'] - df['low']
        
        # Detectar Doji (cuerpo pequeño en relación al rango)
        df['doji'] = (df['body_size'] / df['range_size'] < threshold) & (df['range_size'] > 0)
        
        # Limpiar columnas temporales
        df.drop(['body_size', 'range_size'], axis=1, inplace=True)
        
        return df
    
    def _add_hammer(self, 
                   df: pd.DataFrame, 
                   body_ratio: float = 0.3, 
                   shadow_ratio: float = 2.0) -> pd.DataFrame:
        """
        Añade detector de patrones Hammer y Hanging Man.
        
        Args:
            df: DataFrame con datos
            body_ratio: Ratio máximo del cuerpo respecto al rango total
            shadow_ratio: Ratio mínimo de la sombra inferior respecto al cuerpo
            
        Returns:
            DataFrame con detector de Hammer añadido
        """
        # Comprobar que existen las columnas necesarias
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                self.logger.warning(f"Columna '{col}' no encontrada para detectar Hammer")
                return df
        
        # Calcular tamaños
        df['body_size'] = abs(df['close'] - df['open'])
        df['range_size'] = df['high'] - df['low']
        
        # Calcular sombras
        df['upper_shadow'] = df.apply(
            lambda x: x['high'] - max(x['open'], x['close']), axis=1
        )
        df['lower_shadow'] = df.apply(
            lambda x: min(x['open'], x['close']) - x['low'], axis=1
        )
        
        # Detectar Hammer (cuerpo pequeño, sombra inferior larga, sombra superior corta)
        df['hammer'] = (
            (df['body_size'] / df['range_size'] <= body_ratio) &  # Cuerpo pequeño
            (df['lower_shadow'] / df['body_size'] >= shadow_ratio) &  # Sombra inferior larga
            (df['upper_shadow'] < df['body_size'])  # Sombra superior corta
        )
        
        # Detectar Hanging Man (similar al Hammer pero en tendencia alcista)
        # Necesitaríamos información de tendencia para distinguirlos correctamente
        
        # Limpiar columnas temporales
        df.drop(['body_size', 'range_size', 'upper_shadow', 'lower_shadow'], axis=1, inplace=True)
        
        return df
    
    def _add_engulfing(self, 
                      df: pd.DataFrame) -> pd.DataFrame:
        """
        Añade detector de patrones Engulfing (envolvente).
        
        Args:
            df: DataFrame con datos
            
        Returns:
            DataFrame con detector de Engulfing añadido
        """
        # Comprobar que existen las columnas necesarias
        required_cols = ['open', 'close']
        for col in required_cols:
            if col not in df.columns:
                self.logger.warning(f"Columna '{col}' no encontrada para detectar Engulfing")
                return df
        
        # Calcular direcciones (1 para alcista, -1 para bajista)
        df['direction'] = np.sign(df['close'] - df['open'])
        
        # Detectar patrones Engulfing
        df['bullish_engulfing'] = False
        df['bearish_engulfing'] = False
        
        for i in range(1, len(df)):
            # Patrón alcista: vela anterior bajista, vela actual alcista, 
            # apertura actual por debajo del cierre anterior, cierre actual por encima de la apertura anterior
            if (df['direction'].iloc[i-1] < 0 and 
                df['direction'].iloc[i] > 0 and 
                df['open'].iloc[i] <= df['close'].iloc[i-1] and 
                df['close'].iloc[i] >= df['open'].iloc[i-1]):
                df['bullish_engulfing'].iloc[i] = True
            
            # Patrón bajista: vela anterior alcista, vela actual bajista, 
            # apertura actual por encima del cierre anterior, cierre actual por debajo de la apertura anterior
            elif (df['direction'].iloc[i-1] > 0 and 
                  df['direction'].iloc[i] < 0 and 
                  df['open'].iloc[i] >= df['close'].iloc[i-1] and 
                  df['close'].iloc[i] <= df['open'].iloc[i-1]):
                df['bearish_engulfing'].iloc[i] = True
        
        # Limpiar columnas temporales
        df.drop(['direction'], axis=1, inplace=True)
        
        return df