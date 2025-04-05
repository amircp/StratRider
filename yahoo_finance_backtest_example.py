import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import logging

from connectors.yahoo_connector import YahooFinanceConnector

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Importar componentes del framework

from strategies.example_strategies import MovingAverageCrossover
from engine.backtest_engine import BacktestEngine
from metrics.metrics_calculator import MetricsCalculator
from reporting.report_generator import ReportGenerator

# Importar el framework principal
from main import BacktestFramework


def run_yahoo_finance_example():
    """
    Ejecuta un ejemplo de backtesting usando el conector de Yahoo Finance.
    """
    print("=== EJEMPLO DE BACKTESTING CON YAHOO FINANCE ===")

    # Crear conector para obtener datos
    print("\n1. Configurando conector de Yahoo Finance...")
    connector = YahooFinanceConnector()

    # Listar algunos símbolos disponibles
    available_symbols = connector.get_available_symbols()
    print(f"Símbolos disponibles para ejemplo: {', '.join(available_symbols[:5])}...")

    # Obtener timeframes disponibles
    timeframes = connector.get_available_timeframes()
    print(f"Timeframes disponibles: {', '.join(timeframes)}")

    # Crear framework
    print("\n2. Inicializando framework...")
    framework = BacktestFramework(
        connector=connector,
        initial_capital=10000,
        commission_rate=0.001,  # 0.1%
        position_sizing='percent',
        position_size=0.02,  # Usar 95% del capital por operación
        reports_dir='./yahoo_reports'
    )

    # Definir fechas de backtesting (último año)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    # Definir símbolos y timeframes a probar
    symbols = ['AAPL', 'MSFT']
    timeframe = '1d'  # Usar datos diarios para mayor disponibilidad

    # Crear estrategias a probar
    print("\n3. Configurando estrategias...")
    strategies = [
        MovingAverageCrossover(fast_period=20, slow_period=50, name="MA Crossover (20,50)"),
        MovingAverageCrossover(fast_period=10, slow_period=30, name="MA Crossover (10,30)")
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
                print(f"    Win rate: {result['win_rate']:.1f}%")

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
                'Sharpe': f"{result['sharpe_ratio']:.2f}",
                'Max DD (%)': f"{result['max_drawdown_pct']:.2f}",
                'Win Rate (%)': f"{result['win_rate']:.1f}",
                'Trades': result['trades'].shape[0] if 'trades' in result and not result['trades'].empty else 0
            })

        # Mostrar tabla
        df_results = pd.DataFrame(table_data)
        print(df_results.to_string(index=False))

        # Guardar resultados en CSV
        df_results.to_csv('yahoo_finance_backtest_results.csv', index=False)
        print("\nResultados guardados en 'yahoo_finance_backtest_results.csv'")

    print("\n=== EJEMPLO COMPLETADO ===")
    return results


# Ejecutar ejemplo
if __name__ == "__main__":
    run_yahoo_finance_example()