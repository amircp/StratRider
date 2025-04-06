#!/usr/bin/env python3
import click
import os
import sys
import json
import pandas as pd
from datetime import datetime, timedelta
import importlib
import inspect
import logging
from typing import Dict, List, Any, Optional, Type
import calendar

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('StratRider')

# Añadir el directorio actual al path para poder importar los módulos del framework
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar componentes del framework
from connectors.base_connector import IDataConnector
from connectors.crypto_connector import CryptoConnector
from connectors.yahoo_connector import YahooFinanceConnector
from connectors.database_connector import InfluxDBConnector
from strategies.base_strategy import IStrategy, SignalType
from main import BacktestFramework


# Crear grupo de comandos principal
@click.group()
def cli():
    """StratRider: Framework de Backtesting para Estrategias de Trading."""
    pass


# Comando para crear cosas
@cli.group()
def create():
    """Crear nuevos componentes (estrategias, etc.)."""
    pass


# Comando para listar cosas
@cli.group()
def list():
    """Listar componentes disponibles (estrategias, conectores, etc.)."""
    pass


# Comando para ejecutar cosas
@cli.group()
def run():
    """Ejecutar backtests, optimizaciones, etc."""
    pass


# Comando para optimizar
@cli.group()
def optimize():
    """Ejecutar optimizaciones de estrategias."""
    pass


# Comando para análisis walk-forward
@cli.group()
def walkforward():
    """Ejecutar análisis walk-forward."""
    pass


# Funciones auxiliares

def get_available_strategies() -> Dict[str, Type[IStrategy]]:
    """Obtiene todas las estrategias disponibles en el directorio strategies."""
    strategies = {}
    strategies_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'strategies')

    # Listar archivos Python en el directorio de estrategias
    for filename in os.listdir(strategies_dir):
        if filename.endswith('.py') and filename != '__init__.py' and filename != 'base_strategy.py':
            module_name = filename[:-3]  # Eliminar .py

            try:
                # Importar el módulo
                module = importlib.import_module(f'strategies.{module_name}')

                # Buscar clases que hereden de IStrategy
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and issubclass(obj, IStrategy) and obj != IStrategy:
                        strategies[name] = obj
            except ImportError as e:
                logger.error(f"No se pudo importar el módulo {module_name}: {e}")

    return strategies


def get_available_connectors() -> Dict[str, Type[IDataConnector]]:
    """Obtiene todos los conectores disponibles."""
    connectors = {
        'crypto': CryptoConnector,
        'yahoo': YahooFinanceConnector,
        'influxdb': InfluxDBConnector
    }
    return connectors


def create_strategy_from_template(name: str, template: str) -> str:
    """Crea una nueva estrategia a partir de una plantilla."""
    # Verificar si ya existe una estrategia con ese nombre
    strategies_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'strategies')
    strategy_file = os.path.join(strategies_dir, f"{name.lower()}_strategy.py")

    if os.path.exists(strategy_file):
        return f"Ya existe una estrategia con el nombre {name}"

    # Obtener la plantilla basada en el tipo
    template_code = ""
    class_name = f"{name.replace('_', ' ').title().replace(' ', '')}Strategy"

    if template == "moving_average":
        template_code = f"""import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import talib

from strategies.base_strategy import IStrategy, Signal, SignalType

class {class_name}(IStrategy):
    '''
    Estrategia basada en el cruce de medias móviles.
    '''

    def __init__(self, 
                fast_period: int = 20, 
                slow_period: int = 50,
                name: str = "{name}"):
        '''
        Inicializa la estrategia con los períodos de las medias móviles.

        Args:
            fast_period: Período de la media móvil rápida
            slow_period: Período de la media móvil lenta
            name: Nombre de la estrategia
        '''
        super().__init__(name=name)

        self.parameters = {{
            'fast_period': fast_period,
            'slow_period': slow_period
        }}

        self.data = None
        self.position = 0  # 0: sin posición, 1: long, -1: short

    def initialize(self, data: pd.DataFrame) -> None:
        '''
        Inicializa la estrategia con los datos históricos y calcula indicadores.

        Args:
            data: DataFrame con datos históricos OHLCV
        '''
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
        '''
        Procesa una nueva vela y genera una señal si hay un cruce.

        Args:
            candle: Serie de Pandas con datos de la vela actual

        Returns:
            Señal generada o None si no hay señal
        '''
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
                metadata={{
                    'fast_ma': candle['fast_ma'],
                    'slow_ma': candle['slow_ma']
                }}
            )
            self.signals.append(signal)
            self.position = 1
            return signal

        elif signal_change == -2:  # Cruce bajista (de 1 a -1)
            signal = Signal(
                timestamp=timestamp,
                signal_type=SignalType.SELL,
                price=close_price,
                metadata={{
                    'fast_ma': candle['fast_ma'],
                    'slow_ma': candle['slow_ma']
                }}
            )
            self.signals.append(signal)
            self.position = -1
            return signal

        return None

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        '''
        Establece los parámetros de la estrategia y recalcula indicadores.

        Args:
            parameters: Diccionario con los parámetros
        '''
        super().set_parameters(parameters)

        # Si ya hay datos, recalcular indicadores
        if self.data is not None:
            self.initialize(self.data)
"""
    elif template == "bollinger":
        template_code = f"""import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import talib

from strategies.base_strategy import IStrategy, Signal, SignalType

class {class_name}(IStrategy):
    '''
    Estrategia basada en Bandas de Bollinger.
    '''

    def __init__(self, 
                period: int = 20, 
                deviations: float = 2.0,
                exit_opposite: bool = True,
                name: str = "{name}"):
        '''
        Inicializa la estrategia.

        Args:
            period: Período para las Bandas de Bollinger
            deviations: Número de desviaciones estándar
            exit_opposite: Si salir con señales opuestas
            name: Nombre de la estrategia
        '''
        super().__init__(name=name)

        self.parameters = {{
            'period': period,
            'deviations': deviations,
            'exit_opposite': exit_opposite
        }}

        self.data = None
        self.position = 0  # 0: sin posición, 1: long, -1: short

    def initialize(self, data: pd.DataFrame) -> None:
        '''
        Inicializa la estrategia con los datos históricos y calcula indicadores.

        Args:
            data: DataFrame con datos históricos OHLCV
        '''
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
        self.data['bb_pct_b'] = (self.data['close'] - self.data['bb_lower']) / (self.data['bb_upper'] - self.data['bb_lower'])

        # Calcular cruces con las bandas
        self.data['cross_upper'] = (self.data['close'] > self.data['bb_upper']) & (self.data['close'].shift(1) <= self.data['bb_upper'].shift(1))
        self.data['cross_lower'] = (self.data['close'] < self.data['bb_lower']) & (self.data['close'].shift(1) >= self.data['bb_lower'].shift(1))

        # Resetear señales
        self.signals = []
        self.position = 0

    def process_candle(self, candle: pd.Series) -> Optional[Signal]:
        '''
        Procesa una nueva vela y genera una señal si hay un cruce con las bandas.

        Args:
            candle: Serie de Pandas con datos de la vela actual

        Returns:
            Señal generada o None si no hay señal
        '''
        # AQUÍ DEBES IMPLEMENTAR TU LÓGICA DE TRADING
        # Ejemplo: generar señal cuando el precio cruza las bandas
        timestamp = candle.name
        close_price = candle['close']

        # Comprobar si hay cruce de bandas
        if candle['cross_upper']:
            signal = Signal(
                timestamp=timestamp,
                signal_type=SignalType.BUY,
                price=close_price,
                metadata={{
                    'bb_upper': candle['bb_upper'],
                    'bb_lower': candle['bb_lower']
                }}
            )
            self.signals.append(signal)
            self.position = 1
            return signal

        elif candle['cross_lower']:
            signal = Signal(
                timestamp=timestamp,
                signal_type=SignalType.SELL,
                price=close_price,
                metadata={{
                    'bb_upper': candle['bb_upper'],
                    'bb_lower': candle['bb_lower']
                }}
            )
            self.signals.append(signal)
            self.position = -1
            return signal

        return None

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        '''
        Establece los parámetros de la estrategia y recalcula indicadores.

        Args:
            parameters: Diccionario con los parámetros
        '''
        super().set_parameters(parameters)

        # Si ya hay datos, recalcular indicadores
        if self.data is not None:
            self.initialize(self.data)
"""
    elif template == "rsi":
        template_code = f"""import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import talib

from strategies.base_strategy import IStrategy, Signal, SignalType

class {class_name}(IStrategy):
    '''
    Estrategia basada en el indicador RSI (Relative Strength Index).
    '''

    def __init__(self, 
                period: int = 14,
                overbought: float = 70.0,
                oversold: float = 30.0,
                name: str = "{name}"):
        '''
        Inicializa la estrategia.

        Args:
            period: Período del RSI
            overbought: Nivel de sobrecompra
            oversold: Nivel de sobreventa
            name: Nombre de la estrategia
        '''
        super().__init__(name=name)

        self.parameters = {{
            'period': period,
            'overbought': overbought,
            'oversold': oversold
        }}

        self.data = None
        self.position = 0  # 0: sin posición, 1: long, -1: short

    def initialize(self, data: pd.DataFrame) -> None:
        '''
        Inicializa la estrategia con los datos históricos y calcula indicadores.

        Args:
            data: DataFrame con datos históricos OHLCV
        '''
        self.data = data.copy()

        # Calcular RSI
        self.data['rsi'] = talib.RSI(self.data['close'].values, timeperiod=self.parameters['period'])

        # Detectar cruces con niveles
        self.data['oversold_cross_up'] = (self.data['rsi'] > self.parameters['oversold']) & (self.data['rsi'].shift(1) <= self.parameters['oversold'])
        self.data['overbought_cross_down'] = (self.data['rsi'] < self.parameters['overbought']) & (self.data['rsi'].shift(1) >= self.parameters['overbought'])

        # Resetear señales
        self.signals = []
        self.position = 0

    def process_candle(self, candle: pd.Series) -> Optional[Signal]:
        '''
        Procesa una nueva vela y genera una señal si el RSI cruza los niveles.

        Args:
            candle: Serie de Pandas con datos de la vela actual

        Returns:
            Señal generada o None si no hay señal
        '''
        # Obtener valores relevantes
        timestamp = candle.name
        close_price = candle['close']

        # Verificar si hay un cruce del RSI
        if candle['oversold_cross_up'] and self.position <= 0:
            # Señal de compra: RSI cruza hacia arriba el nivel de sobreventa
            signal = Signal(
                timestamp=timestamp,
                signal_type=SignalType.BUY,
                price=close_price,
                metadata={{
                    'rsi': candle['rsi']
                }}
            )
            self.signals.append(signal)
            self.position = 1
            return signal

        elif candle['overbought_cross_down'] and self.position >= 0:
            # Señal de venta: RSI cruza hacia abajo el nivel de sobrecompra
            signal = Signal(
                timestamp=timestamp,
                signal_type=SignalType.SELL,
                price=close_price,
                metadata={{
                    'rsi': candle['rsi']
                }}
            )
            self.signals.append(signal)
            self.position = -1
            return signal

        return None

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        '''
        Establece los parámetros de la estrategia y recalcula indicadores.

        Args:
            parameters: Diccionario con los parámetros
        '''
        super().set_parameters(parameters)

        # Si ya hay datos, recalcular indicadores
        if self.data is not None:
            self.initialize(self.data)
"""
    elif template == "empty":
        template_code = f"""import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import talib

from strategies.base_strategy import IStrategy, Signal, SignalType

class {class_name}(IStrategy):
    '''
    Descripción de tu estrategia aquí.
    '''

    def __init__(self, name: str = "{name}"):
        '''
        Inicializa la estrategia.

        Args:
            name: Nombre de la estrategia
        '''
        super().__init__(name=name)

        self.parameters = {{
            # Definir los parámetros iniciales aquí
        }}

        self.data = None
        self.position = 0  # 0: sin posición, 1: long, -1: short

    def initialize(self, data: pd.DataFrame) -> None:
        '''
        Inicializa la estrategia con los datos históricos y calcula indicadores.

        Args:
            data: DataFrame con datos históricos OHLCV
        '''
        self.data = data.copy()

        # Calcular indicadores aquí

        # Resetear señales
        self.signals = []
        self.position = 0

    def process_candle(self, candle: pd.Series) -> Optional[Signal]:
        '''
        Procesa una nueva vela y genera una señal si es necesario.

        Args:
            candle: Serie de Pandas con datos de la vela actual

        Returns:
            Señal generada o None si no hay señal
        '''
        # AQUÍ DEBES IMPLEMENTAR TU LÓGICA DE TRADING
        # Obtener valores relevantes
        timestamp = candle.name
        close_price = candle['close']

        # Ejemplo: generar una señal de compra
        # if condición_de_compra:
        #     signal = Signal(
        #         timestamp=timestamp,
        #         signal_type=SignalType.BUY,
        #         price=close_price,
        #         metadata=
        #     )
        #     self.signals.append(signal)
        #     self.position = 1
        #     return signal

        return None

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        '''
        Establece los parámetros de la estrategia y recalcula indicadores.

        Args:
            parameters: Diccionario con los parámetros
        '''
        super().set_parameters(parameters)

        # Si ya hay datos, recalcular indicadores
        if self.data is not None:
            self.initialize(self.data)
"""

    # Escribir el archivo de la estrategia
    with open(strategy_file, 'w') as f:
        f.write(template_code)

    return f"Estrategia {name} creada correctamente en {strategy_file}"


# Implementación de los comandos específicos

@create.command('strategy')
@click.option('--name', required=True, help='Nombre de la estrategia')
@click.option('--template', type=click.Choice(['moving_average', 'bollinger', 'rsi', 'empty']), default='empty',
              help='Plantilla a utilizar')
def create_strategy(name, template):
    """Crear una nueva estrategia basada en una plantilla."""
    result = create_strategy_from_template(name, template)
    click.echo(result)


@list.command('strategies')
def list_strategies():
    """Listar estrategias disponibles."""
    strategies = get_available_strategies()

    if strategies:
        click.echo("Estrategias disponibles:")
        for name, strategy_class in strategies.items():
            click.echo(f"- {name}")
    else:
        click.echo("No se encontraron estrategias.")


@list.command('connectors')
def list_connectors():
    """Listar conectores disponibles."""
    connectors = get_available_connectors()

    if connectors:
        click.echo("Conectores disponibles:")
        for name in connectors.keys():
            click.echo(f"- {name}")
    else:
        click.echo("No se encontraron conectores.")


@run.command('backtest')
@click.option('--strategy', required=True, help='Nombre de la estrategia a utilizar')
@click.option('--connector', type=click.Choice(['crypto', 'yahoo', 'influxdb']), default='crypto',
              help='Tipo de conector a utilizar')
@click.option('--symbol', required=True, help='Símbolo del activo (ej. BTC/USDT, AAPL)')
@click.option('--timeframe', default='1h', help='Intervalo de tiempo (ej. 1h, 1d)')
@click.option('--start-date', help='Fecha de inicio (formato: YYYY-MM-DD)')
@click.option('--end-date', help='Fecha de fin (formato: YYYY-MM-DD)')
@click.option('--capital', type=float, default=10000.0, help='Capital inicial')
@click.option('--commission', type=float, default=0.001, help='Tasa de comisión')
@click.option('--position-size', type=float, default=1.0, help='Tamaño de la posición (porcentaje o unidades)')
@click.option('--position-sizing', type=click.Choice(['fixed', 'percent', 'risk_pct']), default='percent',
              help='Método de dimensionamiento de posiciones')
@click.option('--report-dir', default='./reports', help='Directorio para guardar reportes')
@click.option('--report-template', default=None, help='Plantilla para el reporte (ej. professional)')
@click.option('--exchange', default='binance', help='Exchange para el conector crypto')
@click.option('--api-key', help='API Key para el conector crypto')
@click.option('--api-secret', help='API Secret para el conector crypto')
@click.option('--influxdb-url', help='URL para el conector InfluxDB')
@click.option('--influxdb-token', help='Token para el conector InfluxDB')
@click.option('--influxdb-org', help='Organización para el conector InfluxDB')
@click.option('--influxdb-bucket', help='Bucket para el conector InfluxDB')
def run_backtest(strategy, connector, symbol, timeframe, start_date, end_date, capital, commission, position_size,
                 position_sizing, report_dir, report_template, exchange, api_key, api_secret, influxdb_url,
                 influxdb_token, influxdb_org, influxdb_bucket):
    """Ejecutar un backtest de una estrategia."""
    # Convertir fechas
    if start_date:
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    else:
        start_date = datetime.now() - timedelta(days=180)  # Por defecto, últimos 6 meses

    if end_date:
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    else:
        end_date = datetime.now()

    # Crear conector
    conn = None
    if connector == 'crypto':
        conn = CryptoConnector(exchange_id=exchange, api_key=api_key, api_secret=api_secret)
    elif connector == 'yahoo':
        conn = YahooFinanceConnector()
    elif connector == 'influxdb':
        if not all([influxdb_url, influxdb_token, influxdb_org, influxdb_bucket]):
            click.echo("Error: Se requieren todos los parámetros de InfluxDB (url, token, org, bucket).")
            return
        conn = InfluxDBConnector(url=influxdb_url, token=influxdb_token, org=influxdb_org, bucket=influxdb_bucket)

    if conn is None:
        click.echo(f"Error: No se pudo inicializar el conector {connector}.")
        return

    # Obtener la clase de la estrategia
    strategies = get_available_strategies()
    if strategy not in strategies:
        click.echo(f"Error: Estrategia {strategy} no encontrada.")
        return

    strategy_class = strategies[strategy]

    # Crear instancia de la estrategia
    strategy_instance = strategy_class()

    # Crear framework
    framework = BacktestFramework(
        connector=conn,
        initial_capital=capital,
        commission_rate=commission,
        position_sizing=position_sizing,
        position_size=position_size,
        reports_dir=report_dir,
        report_template=report_template
    )

    # Ejecutar backtest
    try:
        click.echo(f"Ejecutando backtest para {symbol} con {strategy}...")

        # Definir indicadores comunes
        indicators = [
            {'type': 'rsi', 'period': 14},
            {'type': 'macd', 'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
            {'type': 'bollinger', 'period': 20, 'deviations': 2.0}
        ]

        # Ejecutar backtest
        result = framework.run_backtest(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            strategy=strategy_instance,
            indicators=indicators
        )

        # Generar reporte
        report_path = framework.generate_report(result)

        # Mostrar resultados
        click.echo("\n--- RESULTADOS DEL BACKTEST ---")
        click.echo(f"Estrategia: {strategy}")
        click.echo(f"Símbolo: {symbol}")
        click.echo(f"Período: {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}")
        click.echo(f"Capital inicial: ${capital:,.2f}")
        click.echo(f"Capital final: ${result['final_equity']:,.2f}")
        click.echo(f"Retorno total: {result['total_return_pct']:.2f}%")
        click.echo(f"CAGR: {result.get('cagr_pct', 0):.2f}%")
        click.echo(f"Sharpe ratio: {result['sharpe_ratio']:.2f}")
        click.echo(f"Sortino ratio: {result.get('sortino_ratio', 0):.2f}")
        click.echo(f"Drawdown máximo: {result['max_drawdown_pct']:.2f}%")
        click.echo(f"Win rate: {result['win_rate']:.1f}%")
        click.echo(f"Profit factor: {result['profit_factor']:.2f}")
        click.echo(f"Operaciones totales: {result['total_trades']}")
        click.echo(f"Reporte generado: {report_path}")

    except Exception as e:
        click.echo(f"Error al ejecutar backtest: {e}")


@optimize.command('strategy')
@click.option('--strategy', required=True, help='Nombre de la estrategia a optimizar')
@click.option('--connector', type=click.Choice(['crypto', 'yahoo', 'influxdb']), default='crypto',
              help='Tipo de conector a utilizar')
@click.option('--symbol', required=True, help='Símbolo del activo (ej. BTC/USDT, AAPL)')
@click.option('--timeframe', default='1h', help='Intervalo de tiempo (ej. 1h, 1d)')
@click.option('--start-date', help='Fecha de inicio (formato: YYYY-MM-DD)')
@click.option('--end-date', help='Fecha de fin (formato: YYYY-MM-DD)')
@click.option('--param-grid', required=True, help='Rejilla de parámetros en formato JSON')
@click.option('--metric', default='sharpe_ratio', help='Métrica a optimizar')
@click.option('--parallel/--no-parallel', default=False, help='Ejecutar en paralelo')
@click.option('--n-jobs', type=int, default=-1, help='Número de trabajos paralelos (-1 para todos)')
@click.option('--output-file', default='optimization_results.csv', help='Archivo para guardar resultados')
def optimize_strategy(strategy, connector, symbol, timeframe, start_date, end_date, param_grid, metric, parallel,
                      n_jobs, output_file):
    """Optimizar parámetros de una estrategia."""
    # Convertir fechas
    if start_date:
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    else:
        start_date = datetime.now() - timedelta(days=180)  # Por defecto, últimos 6 meses

    if end_date:
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    else:
        end_date = datetime.now()

    # Parsear param_grid
    try:
        param_grid_dict = json.loads(param_grid)
    except json.JSONDecodeError:
        click.echo("Error: param-grid debe ser un JSON válido.")
        return

    # Crear conector
    conn = None
    if connector == 'crypto':
        conn = CryptoConnector(exchange_id='binance')
    elif connector == 'yahoo':
        conn = YahooFinanceConnector()
    elif connector == 'influxdb':
        click.echo("Error: Para el conector InfluxDB, use el comando completo con todos los parámetros.")
        return

    if conn is None:
        click.echo(f"Error: No se pudo inicializar el conector {connector}.")
        return

    # Obtener la clase de la estrategia
    strategies = get_available_strategies()
    if strategy not in strategies:
        click.echo(f"Error: Estrategia {strategy} no encontrada.")
        return

    strategy_class = strategies[strategy]

    # Crear framework
    framework = BacktestFramework(
        connector=conn,
        initial_capital=10000,
        commission_rate=0.001,
        position_sizing='percent',
        position_size=0.95
    )

    # Ejecutar optimización
    try:
        click.echo(f"Optimizando parámetros para {symbol} con {strategy}...")

        # Ejecutar optimización
        results = framework.run_optimization(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            strategy_class=strategy_class,
            param_grid=param_grid_dict,
            metric=metric,
            parallel=parallel,
            n_jobs=n_jobs
        )

        # Guardar resultados
        results.to_csv(output_file, index=False)

        # Mostrar mejores combinaciones
        click.echo("\n--- RESULTADOS DE LA OPTIMIZACIÓN ---")
        click.echo("Mejores combinaciones de parámetros:")
        click.echo(results.head(10).to_string())
        click.echo(f"\nResultados guardados en {output_file}")

    except Exception as e:
        click.echo(f"Error al ejecutar optimización: {e}")


@walkforward.command('analyze')
@click.option('--strategy', required=True, help='Nombre de la estrategia a analizar')
@click.option('--connector', type=click.Choice(['crypto', 'yahoo', 'influxdb']), default='crypto',
              help='Tipo de conector a utilizar')
@click.option('--symbol', required=True, help='Símbolo del activo (ej. BTC/USDT, AAPL)')
@click.option('--timeframe', default='1h', help='Intervalo de tiempo (ej. 1h, 1d)')
@click.option('--start-date', help='Fecha de inicio (formato: YYYY-MM-DD)')
@click.option('--end-date', help='Fecha de fin (formato: YYYY-MM-DD)')
@click.option('--param-grid', required=True, help='Rejilla de parámetros en formato JSON')
@click.option('--train-size', type=int, default=180, help='Tamaño del período de entrenamiento en días')
@click.option('--test-size', type=int, default=60, help='Tamaño del período de prueba en días')
@click.option('--metric', default='sharpe_ratio', help='Métrica a optimizar')
@click.option('--output-file', default='walkforward_results.csv', help='Archivo para guardar resultados')
def walkforward_analyze(strategy, connector, symbol, timeframe, start_date, end_date, param_grid, train_size, test_size,
                        metric, output_file):
    """Ejecutar análisis walk-forward de una estrategia."""
    # Convertir fechas
    if start_date:
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    else:
        start_date = datetime.now() - timedelta(days=365)  # Por defecto, último año

    if end_date:
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    else:
        end_date = datetime.now()

    # Parsear param_grid
    try:
        param_grid_dict = json.loads(param_grid)
    except json.JSONDecodeError:
        click.echo("Error: param-grid debe ser un JSON válido.")
        return

    # Crear conector
    conn = None
    if connector == 'crypto':
        conn = CryptoConnector(exchange_id='binance')
    elif connector == 'yahoo':
        conn = YahooFinanceConnector()
    elif connector == 'influxdb':
        click.echo("Error: Para el conector InfluxDB, use el comando completo con todos los parámetros.")
        return

    if conn is None:
        click.echo(f"Error: No se pudo inicializar el conector {connector}.")
        return

    # Obtener la clase de la estrategia
    strategies = get_available_strategies()
    if strategy not in strategies:
        click.echo(f"Error: Estrategia {strategy} no encontrada.")
        return

    strategy_class = strategies[strategy]

    # Crear framework
    framework = BacktestFramework(
        connector=conn,
        initial_capital=10000,
        commission_rate=0.001,
        position_sizing='percent',
        position_size=0.95
    )

    # Ejecutar análisis walk-forward
    try:
        click.echo(f"Ejecutando análisis walk-forward para {symbol} con {strategy}...")

        # Ejecutar análisis
        results = framework.walk_forward_analysis(
            symbol=symbol,
            timeframe=timeframe,
            strategy_class=strategy_class,
            param_grid=param_grid_dict,
            start_date=start_date,
            end_date=end_date,
            train_size=train_size,
            test_size=test_size,
            metric=metric
        )

        # Guardar resultados
        results['results_df'].to_csv(output_file, index=False)

        # Mostrar estadísticas
        click.echo("\n--- RESULTADOS DEL ANÁLISIS WALK-FORWARD ---")
        click.echo("Estadísticas del análisis:")
        for key, value in results['stats'].items():
            if key != 'avg_params':
                click.echo(f"  {key}: {value}")

        click.echo("\nParámetros promedio:")
        for param, value in results['stats']['avg_params'].items():
            click.echo(f"  {param}: {value}")

        click.echo("\nResultados por período:")
        click.echo(
            results['results_df'][['period', 'test_return', 'test_sharpe', 'test_win_rate', 'test_trades']].to_string())

        click.echo(f"\nResultados guardados en {output_file}")

    except Exception as e:
        click.echo(f"Error al ejecutar análisis walk-forward: {e}")


if __name__ == '__main__':
    cli()