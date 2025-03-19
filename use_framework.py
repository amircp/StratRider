import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Importar componentes del framework
from connectors.crypto_connector import CryptoConnector
from strategies.example_strategies import MovingAverageCrossover
from strategies.bollinger_strategy import BollingerBreakoutStrategy
from engine.backtest_engine import BacktestEngine
from metrics.metrics_calculator import MetricsCalculator
from reporting.report_generator import ReportGenerator
from data.data_processor import DataProcessor

# Importar el framework principal
from main import BacktestFramework

def run_example():
    """
    Ejecuta un ejemplo completo del framework de backtesting.
    """
    print("=== FRAMEWORK DE BACKTESTING DE CRIPTOMONEDAS ===")
    print("Iniciando ejemplo de backtesting...")
    
    # Crear conector para obtener datos
    print("\n1. Configurando conector de datos...")
    connector = CryptoConnector(exchange_id='binance')
    
    # Crear framework
    print("\n2. Inicializando framework...")
    framework = BacktestFramework(
        connector=connector,
        initial_capital=10000,
        commission_rate=0.001,  # 0.1%
        position_sizing='percent',
        position_size=0.95,  # Usar 95% del capital por operación
        reports_dir='./reports'
    )
    
    # Definir fechas de backtesting
    start_date = datetime.now() - timedelta(days=180)  # Últimos 6 meses
    end_date = datetime.now()
    
    # Definir símbolos y timeframes a probar
    symbols = ['BTC/USDT', 'ETH/USDT']
    timeframe = '1h'  # 1 hora
    
    # Crear estrategias a probar
    print("\n3. Creando estrategias...")
    strategies = [
        MovingAverageCrossover(fast_period=20, slow_period=50, name="MA Crossover (20,50)"),
        MovingAverageCrossover(fast_period=10, slow_period=30, name="MA Crossover (10,30)"),
        BollingerBreakoutStrategy(period=20, deviations=2.0, name="Bollinger Breakout (20,2)")
    ]
    
    # Definir indicadores adicionales
    indicators = [
        {'type': 'rsi', 'period': 14},
        {'type': 'macd', 'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
        {'type': 'bollinger', 'period': 20, 'deviations': 2.0}
    ]
    
    # Ejecutar backtests
    results = []
    
    print("\n4. Ejecutando backtests...")
    for symbol in symbols:
        print(f"\nAnalizando {symbol}...")
        
        for strategy in strategies:
            print(f"  - Estrategia: {strategy.name}")
            
            # Ejecutar backtest
            try:
                result = framework.run_backtest(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    strategy=strategy,
                    indicators=indicators
                )
                
                # Añadir símbolo a resultados
                result['symbol'] = symbol
                results.append(result)
                
                # Generar reporte
                report_path = framework.generate_report(result)
                print(f"    Reporte generado: {report_path}")
                
                # Mostrar métricas principales
                print(f"    Retorno total: {result['total_return_pct']:.2f}%")
                print(f"    Sharpe ratio: {result['sharpe_ratio']:.2f}")
                print(f"    Drawdown máximo: {result['max_drawdown_pct']:.2f}%")
                print(f"    Ratio ganancia/pérdida: {result['avg_win_loss_ratio']:.2f}")
                print(f"    Operaciones: {result['total_trades']} (Ganadoras: {result['win_rate']:.1f}%)")
                
            except Exception as e:
                print(f"    Error al ejecutar backtest: {e}")
    
    # Comparar resultados
    print("\n5. Comparación de resultados:")
    if results:
        # Crear tabla de resultados
        table_data = []
        
        for result in results:
            table_data.append({
                'Symbol': result['symbol'],
                'Strategy': result['strategy_name'],
                'Return (%)': f"{result['total_return_pct']:.2f}",
                'CAGR (%)': f"{result['cagr_pct']:.2f}",
                'Sharpe': f"{result['sharpe_ratio']:.2f}",
                'Max DD (%)': f"{result['max_drawdown_pct']:.2f}",
                'Win Rate (%)': f"{result['win_rate']:.1f}",
                'Profit Factor': f"{result['profit_factor']:.2f}",
                'Trades': result['total_trades']
            })
        
        # Mostrar tabla
        df_results = pd.DataFrame(table_data)
        print(df_results.to_string(index=False))
        
        # Guardar resultados en CSV
        df_results.to_csv('backtest_results_comparison.csv', index=False)
        print("\nResultados guardados en 'backtest_results_comparison.csv'")
    
    print("\n=== EJEMPLO COMPLETADO ===")
    return results

def run_optimization_example():
    """
    Ejecuta un ejemplo de optimización de parámetros.
    """
    print("=== OPTIMIZACIÓN DE PARÁMETROS ===")
    
    # Crear conector para obtener datos
    connector = CryptoConnector(exchange_id='binance')
    
    # Crear framework
    framework = BacktestFramework(
        connector=connector,
        initial_capital=10000,
        commission_rate=0.001
    )
    
    # Definir fechas y símbolo
    start_date = datetime.now() - timedelta(days=180)
    end_date = datetime.now()
    symbol = 'BTC/USDT'
    timeframe = '1h'
    
    # Definir rejilla de parámetros
    param_grid = {
        'fast_period': [5, 10, 15, 20],
        'slow_period': [30, 40, 50, 60]
    }
    
    print(f"\nOptimizando parámetros para {symbol}...")
    
    # Ejecutar optimización
    try:
        results = framework.run_optimization(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            strategy_class=MovingAverageCrossover,
            param_grid=param_grid,
            metric='sharpe_ratio'
        )
        
        # Mostrar mejores combinaciones
        print("\nMejores combinaciones de parámetros:")
        print(results.head(10).to_string())
        
        # Guardar resultados
        results.to_csv('optimization_results.csv', index=False)
        print("\nResultados guardados en 'optimization_results.csv'")
        
        # Visualizar resultados en 3D
        try:
            from mpl_toolkits.mplot3d import Axes3D
            from matplotlib import cm
            
            # Crear pivot table para visualización
            pivot = results.pivot_table(
                values='sharpe_ratio', 
                index='fast_period', 
                columns='slow_period'
            )
            
            # Crear figura
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Preparar datos
            x, y = np.meshgrid(pivot.columns, pivot.index)
            z = pivot.values
            
            # Crear superficie
            surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
            
            # Añadir etiquetas
            ax.set_xlabel('Slow Period')
            ax.set_ylabel('Fast Period')
            ax.set_zlabel('Sharpe Ratio')
            ax.set_title('Optimización de Parámetros')
            
            # Añadir barra de color
            fig.colorbar(surf, shrink=0.5, aspect=5)
            
            # Guardar figura
            plt.savefig('optimization_results.png')
            print("Gráfico guardado en 'optimization_results.png'")
            
        except Exception as e:
            print(f"No se pudo crear visualización 3D: {e}")
    
    except Exception as e:
        print(f"Error al ejecutar optimización: {e}")
    
    print("\n=== OPTIMIZACIÓN COMPLETADA ===")

def run_walkforward_example():
    """
    Ejecuta un ejemplo de análisis walk-forward.
    """
    print("=== ANÁLISIS WALK-FORWARD ===")
    
    # Crear conector para obtener datos
    connector = CryptoConnector(exchange_id='binance')
    
    # Crear framework
    framework = BacktestFramework(
        connector=connector,
        initial_capital=10000,
        commission_rate=0.001
    )
    
    # Definir fechas y símbolo
    start_date = datetime.now() - timedelta(days=365)  # Último año
    end_date = datetime.now()
    symbol = 'BTC/USDT'
    timeframe = '1h'
    
    # Definir rejilla de parámetros
    param_grid = {
        'fast_period': [5, 10, 15, 20],
        'slow_period': [30, 40, 50, 60]
    }
    
    print(f"\nEjecutando análisis walk-forward para {symbol}...")
    
    # Ejecutar análisis walk-forward
    try:
        results = framework.walk_forward_analysis(
            symbol=symbol,
            timeframe=timeframe,
            strategy_class=MovingAverageCrossover,
            param_grid=param_grid,
            start_date=start_date,
            end_date=end_date,
            train_size=90,  # 90 días de entrenamiento
            test_size=30,   # 30 días de prueba
            metric='sharpe_ratio'
        )
        
        # Mostrar estadísticas
        print("\nEstadísticas del análisis walk-forward:")
        for key, value in results['stats'].items():
            if key != 'avg_params':
                print(f"  {key}: {value}")
        
        print("\nParámetros promedio:")
        for param, value in results['stats']['avg_params'].items():
            print(f"  {param}: {value}")
        
        # Mostrar resultados por período
        print("\nResultados por período:")
        print(results['results_df'][['period', 'test_return', 'test_sharpe', 'test_win_rate', 'test_trades']].to_string())
        
        # Guardar resultados
        results['results_df'].to_csv('walkforward_results.csv', index=False)
        print("\nResultados guardados en 'walkforward_results.csv'")
        
    except Exception as e:
        print(f"Error al ejecutar análisis walk-forward: {e}")
    
    print("\n=== ANÁLISIS WALK-FORWARD COMPLETADO ===")

if __name__ == "__main__":
    # Ejecutar ejemplo completo
    results = run_example()
    
    # Ejecutar ejemplo de optimización
    # run_optimization_example()
    
    # Ejecutar ejemplo de análisis walk-forward
    # run_walkforward_example()