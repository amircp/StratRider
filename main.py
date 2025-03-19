import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import logging
import os
pd.set_option('future.no_silent_downcasting', True)
# Importar componentes del framework
from connectors.base_connector import IDataConnector
from data.data_processor import DataProcessor
from strategies.base_strategy import IStrategy
from engine.backtest_engine import BacktestEngine
from metrics.metrics_calculator import MetricsCalculator
from reporting.report_generator import ReportGenerator

class BacktestFramework:
    """
    Framework principal para backtesting de estrategias.
    """
    
    def __init__(self,
                connector: IDataConnector,
                initial_capital: float = 10000.0,
                commission_rate: float = 0.001,
                slippage: float = 0.0,
                position_sizing: str = 'percent',
                position_size: float = 1.0,
                max_open_positions: int = 1,
                reports_dir: str = "./reports",
                verbose: bool = True):
        """
        Inicializa el framework de backtesting.
        
        Args:
            connector: Conector de datos
            initial_capital: Capital inicial
            commission_rate: Tasa de comisión
            slippage: Deslizamiento
            position_sizing: Método de dimensionamiento de posiciones
            position_size: Tamaño de la posición
            max_open_positions: Número máximo de posiciones abiertas
            reports_dir: Directorio para guardar reportes
            verbose: Si mostrar logs detallados
        """
        self.connector = connector
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.position_sizing = position_sizing
        self.position_size = position_size
        self.max_open_positions = max_open_positions
        self.reports_dir = reports_dir
        self.verbose = verbose
        
        # Inicializar componentes
        self.processor = DataProcessor()
        self.engine = BacktestEngine(
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            slippage=slippage,
            position_sizing=position_sizing,
            position_size=position_size,
            max_open_positions=max_open_positions
        )
        self.metrics_calculator = MetricsCalculator()
        self.report_generator = ReportGenerator(output_dir=reports_dir)
        
        # Configurar logging
        logging_level = logging.INFO if verbose else logging.WARNING
        logging.basicConfig(
            level=logging_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('BacktestFramework')
    
    def run_backtest(self,
                    symbol: str,
                    timeframe: str,
                    start_date: Union[str, datetime],
                    end_date: Union[str, datetime],
                    strategy: IStrategy,
                    indicators: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Ejecuta un backtest completo.
        
        Args:
            symbol: Símbolo del activo
            timeframe: Intervalo de tiempo
            start_date: Fecha de inicio
            end_date: Fecha de fin
            strategy: Estrategia a evaluar
            indicators: Lista de indicadores a añadir (opcional)
            
        Returns:
            Diccionario con los resultados del backtest
        """
        # Convertir fechas si son strings
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Log de inicio
        self.logger.info(f"Iniciando backtest para {symbol} en {timeframe} desde {start_date.strftime('%Y-%m-%d')} hasta {end_date.strftime('%Y-%m-%d')}")
        
        # Obtener datos históricos
        try:
            self.logger.info(f"Obteniendo datos históricos para {symbol}...")
            data = self.connector.get_historical_data(symbol, timeframe, start_date, end_date)
            self.logger.info(f"Datos obtenidos: {len(data)} registros")
        except Exception as e:
            self.logger.error(f"Error al obtener datos: {e}")
            raise
        
        # Procesar datos
        if indicators:
            self.logger.info("Procesando datos y añadiendo indicadores...")
            data = self.processor.add_indicators(data, indicators)
        
        # Ejecutar backtest
        self.logger.info(f"Ejecutando backtest con estrategia {strategy.name}...")
        backtest_result = self.engine.run_backtest(data, strategy)
        
        # Calcular métricas
        self.logger.info("Calculando métricas...")
        returns = backtest_result['result_data']['returns'].dropna()
        metrics = self.metrics_calculator.calculate_all_metrics(
            backtest_result['trades'],
            backtest_result['equity_curve'],
            returns
        )
        
        # Combinar resultados y métricas
        result = {**backtest_result, **metrics}
        
        self.logger.info(f"Backtest completado. Retorno total: {metrics['total_return_pct']:.2f}%, Win rate: {metrics['win_rate']:.2f}%")
        
        return result
    
    def generate_report(self,
                       backtest_result: Dict[str, Any],
                       output_file: Optional[str] = None) -> str:
        """
        Genera un reporte HTML con los resultados del backtest.
        
        Args:
            backtest_result: Resultados del backtest
            output_file: Nombre del archivo de salida (opcional)
            
        Returns:
            Ruta al archivo HTML generado
        """
        self.logger.info("Generando reporte HTML...")
        
        # Separar métricas y resultados
        metrics = {k: v for k, v in backtest_result.items() if k not in ['equity_curve', 'trades', 'result_data']}
        
        # Generar reporte
        report_path = self.report_generator.generate_html_report(backtest_result, metrics, output_file)
        
        self.logger.info(f"Reporte generado: {report_path}")
        
        return report_path
    
    def run_optimization(self,
                        symbol: str,
                        timeframe: str,
                        start_date: Union[str, datetime],
                        end_date: Union[str, datetime],
                        strategy_class,
                        param_grid: Dict[str, List[Any]],
                        metric: str = 'sharpe_ratio',
                        parallel: bool = False,
                        n_jobs: int = -1) -> pd.DataFrame:
        """
        Ejecuta una optimización de parámetros para una estrategia.
        
        Args:
            symbol: Símbolo del activo
            timeframe: Intervalo de tiempo
            start_date: Fecha de inicio
            end_date: Fecha de fin
            strategy_class: Clase de la estrategia a optimizar
            param_grid: Rejilla de parámetros a probar
            metric: Métrica a optimizar
            parallel: Si ejecutar en paralelo
            n_jobs: Número de trabajos paralelos (-1 para usar todos los núcleos)
            
        Returns:
            DataFrame con los resultados de la optimización
        """
        # Convertir fechas si son strings
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        self.logger.info(f"Iniciando optimización para {symbol} en {timeframe}...")
        
        # Obtener datos históricos (una sola vez)
        try:
            self.logger.info(f"Obteniendo datos históricos para {symbol}...")
            data = self.connector.get_historical_data(symbol, timeframe, start_date, end_date)
            self.logger.info(f"Datos obtenidos: {len(data)} registros")
        except Exception as e:
            self.logger.error(f"Error al obtener datos: {e}")
            raise
        
        # Generar todas las combinaciones de parámetros
        from itertools import product
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        self.logger.info(f"Probando {len(combinations)} combinaciones de parámetros...")
        
        # Función para evaluar una combinación de parámetros
        def evaluate_params(params):
            # Crear instancia de estrategia con los parámetros
            param_dict = dict(zip(param_names, params))
            strategy = strategy_class(**param_dict)
            
            # Ejecutar backtest
            backtest_result = self.engine.run_backtest(data, strategy)
            
            # Calcular métricas
            returns = backtest_result['result_data']['returns'].dropna()
            metrics = self.metrics_calculator.calculate_all_metrics(
                backtest_result['trades'],
                backtest_result['equity_curve'],
                returns
            )
            
            # Crear resultado
            result = {
                **param_dict,
                'trades': len(backtest_result['trades']),
                'total_return': metrics['total_return_pct'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'sortino_ratio': metrics['sortino_ratio'],
                'max_drawdown': metrics['max_drawdown_pct'],
                'win_rate': metrics['win_rate'],
                'profit_factor': metrics['profit_factor'],
                'expectancy': metrics['expectancy'],
                'optimization_metric': metrics[metric]
            }
            
            return result
        
        # Ejecutar evaluaciones
        results = []
        
        if parallel:
            # Ejecución paralela
            from joblib import Parallel, delayed
            
            results = Parallel(n_jobs=n_jobs)(
                delayed(evaluate_params)(params) for params in combinations
            )
        else:
            # Ejecución secuencial
            total_combos = len(combinations)
            
            for i, params in enumerate(combinations):
                self.logger.info(f"Evaluando combinación {i+1}/{total_combos}...")
                results.append(evaluate_params(params))
        
        # Convertir a DataFrame
        results_df = pd.DataFrame(results)
        
        # Ordenar por la métrica de optimización
        results_df.sort_values(by='optimization_metric', ascending=False, inplace=True)
        
        self.logger.info("Optimización completada.")
        
        return results_df
    
    def walk_forward_analysis(self,
                             symbol: str,
                             timeframe: str,
                             strategy_class,
                             param_grid: Dict[str, List[Any]],
                             start_date: Union[str, datetime],
                             end_date: Union[str, datetime],
                             train_size: int = 180,
                             test_size: int = 60,
                             metric: str = 'sharpe_ratio') -> Dict[str, Any]:
        """
        Realiza un análisis walk-forward.
        
        Args:
            symbol: Símbolo del activo
            timeframe: Intervalo de tiempo
            strategy_class: Clase de la estrategia a optimizar
            param_grid: Rejilla de parámetros a probar
            start_date: Fecha de inicio
            end_date: Fecha de fin
            train_size: Tamaño del período de entrenamiento en días
            test_size: Tamaño del período de prueba en días
            metric: Métrica a optimizar
            
        Returns:
            Diccionario con los resultados del análisis
        """
        # Convertir fechas si son strings
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        self.logger.info(f"Iniciando análisis walk-forward para {symbol} en {timeframe}...")
        
        # Obtener datos históricos completos
        try:
            self.logger.info(f"Obteniendo datos históricos para {symbol}...")
            full_data = self.connector.get_historical_data(symbol, timeframe, start_date, end_date)
            self.logger.info(f"Datos obtenidos: {len(full_data)} registros")
        except Exception as e:
            self.logger.error(f"Error al obtener datos: {e}")
            raise
        
        # Crear períodos de entrenamiento y prueba
        periods = []
        current_date = start_date
        
        while current_date + pd.Timedelta(days=train_size+test_size) <= end_date:
            train_end = current_date + pd.Timedelta(days=train_size)
            test_end = train_end + pd.Timedelta(days=test_size)
            
            periods.append({
                'train_start': current_date,
                'train_end': train_end,
                'test_start': train_end,
                'test_end': test_end
            })
            
            current_date = train_end
        
        # Ejecutar análisis para cada período
        results = []
        
        for i, period in enumerate(periods):
            self.logger.info(f"Procesando período {i+1}/{len(periods)}...")
            
            # Obtener datos de entrenamiento
            train_data = full_data.loc[period['train_start']:period['train_end']]
            
            # Optimizar en datos de entrenamiento
            self.logger.info("Optimizando parámetros en datos de entrenamiento...")
            
            best_params = None
            best_metric_value = float('-inf')
            
            # Generar todas las combinaciones de parámetros
            from itertools import product
            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())
            combinations = list(product(*param_values))
            
            for params in combinations:
                param_dict = dict(zip(param_names, params))
                strategy = strategy_class(**param_dict)
                
                # Ejecutar backtest en datos de entrenamiento
                backtest_result = self.engine.run_backtest(train_data, strategy)
                
                # Calcular métricas
                returns = backtest_result['result_data']['returns'].dropna()
                metrics = self.metrics_calculator.calculate_all_metrics(
                    backtest_result['trades'],
                    backtest_result['equity_curve'],
                    returns
                )
                
                # Comprobar si es mejor
                if metrics[metric] > best_metric_value:
                    best_metric_value = metrics[metric]
                    best_params = param_dict
            
            # Obtener datos de prueba
            test_data = full_data.loc[period['test_start']:period['test_end']]
            
            # Evaluar en datos de prueba con los mejores parámetros
            self.logger.info(f"Evaluando en datos de prueba con parámetros: {best_params}")
            strategy = strategy_class(**best_params)
            
            # Ejecutar backtest en datos de prueba
            backtest_result = self.engine.run_backtest(test_data, strategy)
            
            # Calcular métricas
            returns = backtest_result['result_data']['returns'].dropna()
            metrics = self.metrics_calculator.calculate_all_metrics(
                backtest_result['trades'],
                backtest_result['equity_curve'],
                returns
            )
            
            # Guardar resultados
            period_result = {
                'period': i+1,
                'train_start': period['train_start'],
                'train_end': period['train_end'],
                'test_start': period['test_start'],
                'test_end': period['test_end'],
                'params': best_params,
                'train_metric': best_metric_value,
                'test_metric': metrics[metric],
                'test_return': metrics['total_return_pct'],
                'test_sharpe': metrics['sharpe_ratio'],
                'test_max_dd': metrics['max_drawdown_pct'],
                'test_win_rate': metrics['win_rate'],
                'test_trades': len(backtest_result['trades'])
            }
            
            results.append(period_result)
        
        # Convertir a DataFrame
        results_df = pd.DataFrame(results)
        
        # Calcular estadísticas agregadas
        aggregate_stats = {
            'total_periods': len(results),
            'avg_test_return': results_df['test_return'].mean(),
            'avg_test_sharpe': results_df['test_sharpe'].mean(),
            'avg_test_max_dd': results_df['test_max_dd'].mean(),
            'avg_test_win_rate': results_df['test_win_rate'].mean(),
            'positive_periods': (results_df['test_return'] > 0).sum(),
            'negative_periods': (results_df['test_return'] <= 0).sum(),
            'avg_params': {param: results_df['params'].apply(lambda x: x[param]).mean() for param in param_names}
        }
        
        self.logger.info("Análisis walk-forward completado.")
        
        return {
            'periods': results,
            'stats': aggregate_stats,
            'results_df': results_df
        }