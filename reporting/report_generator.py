import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional
import os
import json
from datetime import datetime
import base64
import io
import calendar
from jinja2 import Template

class ReportGenerator:
    """
    Generador de reportes HTML para backtesting de estrategias.
    """
    
    def __init__(self, 
                template_path: Optional[str] = None,
                output_dir: str = "./reports"):
        """
        Inicializa el generador de reportes.
        
        Args:
            template_path: Ruta a la plantilla HTML (opcional)
            output_dir: Directorio para guardar los reportes
        """
        self.template_path = template_path
        self.output_dir = output_dir
        
        # Crear directorio si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Plantilla HTML por defecto
        self.default_template = """
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{{ title }}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                }
                .header {
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    margin-bottom: 20px;
                    border-radius: 5px;
                }
                .section {
                    margin-bottom: 30px;
                    background: white;
                    padding: 20px;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }
                .metrics-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                    gap: 15px;
                }
                .metric-card {
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                }
                .metric-title {
                    font-size: 14px;
                    color: #666;
                    margin-bottom: 5px;
                }
                .metric-value {
                    font-size: 20px;
                    font-weight: bold;
                    color: #2c3e50;
                }
                .metric-positive {
                    color: #27ae60;
                }
                .metric-negative {
                    color: #e74c3c;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }
                th, td {
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }
                th {
                    background-color: #f8f9fa;
                    font-weight: bold;
                }
                tr:hover {
                    background-color: #f5f5f5;
                }
                .chart-container {
                    height: 450px;
                    margin-bottom: 30px;
                }
                .trades-table-container {
                    max-height: 400px;
                    overflow-y: auto;
                }
                .summary-stats {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    margin-bottom: 20px;
                }
                .summary-stat {
                    flex: 1;
                    min-width: 150px;
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    text-align: center;
                }
                .summary-value {
                    font-size: 24px;
                    font-weight: bold;
                }
                .positive {
                    color: #27ae60;
                }
                .negative {
                    color: #e74c3c;
                }
                .neutral {
                    color: #2c3e50;
                }
                .footer {
                    text-align: center;
                    margin-top: 30px;
                    padding: 20px;
                    font-size: 14px;
                    color: #666;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{{ title }}</h1>
                    <p>{{ subtitle }}</p>
                </div>
                
                <div class="section">
                    <h2>Resumen</h2>
                    <div class="summary-stats">
                        <div class="summary-stat">
                            <div class="summary-title">Retorno Total</div>
                            <div class="summary-value {{ 'positive' if metrics.net_profit_pct > 0 else 'negative' }}">
                                {{ "%.2f" % metrics.net_profit_pct }}%
                            </div>
                        </div>
                        <div class="summary-stat">
                            <div class="summary-title">CAGR</div>
                            <div class="summary-value {{ 'positive' if metrics.cagr_pct > 0 else 'negative' }}">
                                {{ "%.2f" % metrics.cagr_pct }}%
                            </div>
                        </div>
                        <div class="summary-stat">
                            <div class="summary-title">Sharpe Ratio</div>
                            <div class="summary-value {{ 'positive' if metrics.sharpe_ratio > 1 else 'neutral' }}">
                                {{ "%.2f" % metrics.sharpe_ratio }}
                            </div>
                        </div>
                        <div class="summary-stat">
                            <div class="summary-title">Max Drawdown</div>
                            <div class="summary-value negative">
                                {{ "%.2f" % metrics.max_drawdown_pct }}%
                            </div>
                        </div>
                        <div class="summary-stat">
                            <div class="summary-title">Win Rate</div>
                            <div class="summary-value neutral">
                                {{ "%.2f" % metrics.win_rate }}%
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Curva de Equidad</h2>
                    <div class="chart-container" id="equity-chart"></div>
                </div>
                
                <div class="section">
                    <h2>Drawdown</h2>
                    <div class="chart-container" id="drawdown-chart"></div>
                </div>
                
                <div class="section">
                    <h2>Distribución de Rendimientos</h2>
                    <div class="chart-container" id="returns-distribution-chart"></div>
                </div>
                
                <div class="section">
                    <h2>Rendimientos Mensuales</h2>
                    <div class="chart-container" id="monthly-returns-chart"></div>
                </div>
                
                <div class="section">
                    <h2>Métricas Detalladas</h2>
                    <div class="metrics-grid">
                        {% for key, value in metrics.items() %}
                        {% if key not in ['equity_curve', 'trades', 'result_data'] %}
                        <div class="metric-card">
                            <div class="metric-title">{{ key|replace('_', ' ')|title }}</div>
                            <div class="metric-value">
                            {% if 'pct' in key or 'rate' in key %}
                                {{ "%.2f" % value }}%
                            {% elif 'ratio' in key or value is number %}
                                {{ "%.2f" % value }}
                            {% else %}
                                {{ value }}
                            {% endif %}
                            </div>
                        </div>
                        {% endif %}
                        {% endfor %}
                    </div>
                </div>
                
                <div class="section">
                    <h2>Historial de Operaciones</h2>
                    <div class="trades-table-container">
                        <table id="trades-table">
                            <thead>
                                <tr>
                                    <th>Entrada</th>
                                    <th>Tipo</th>
                                    <th>Precio Entrada</th>
                                    <th>Salida</th>
                                    <th>Precio Salida</th>
                                    <th>P/L</th>
                                    <th>P/L %</th>
                                    <th>Duración</th>
                                </tr>
                            </thead>
                            <tbody>
                                {% for trade in trades %}
                                <tr class="{{ 'positive' if trade.profit_loss > 0 else 'negative' }}">
                                    <td>{{ trade.entry_time }}</td>
                                    <td>{{ trade.entry_type }}</td>
                                    <td>{{ "%.2f" % trade.entry_price }}</td>
                                    <td>{{ trade.exit_time }}</td>
                                    <td>{{ "%.2f" % trade.exit_price }}</td>
                                    <td>{{ "%.2f" % trade.profit_loss }}</td>
                                    <td>{{ "%.2f" % trade.profit_loss_pct }}%</td>
                                    <td>{{ trade.duration_str }}</td>
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                </div>
                
                <div class="footer">
                    <p>Generado el {{ generation_time }} | Estrategia: {{ strategy_name }}</p>
                </div>
            </div>
            
            <script>
                // Plotly charts
                {{ equity_chart_js }}
                {{ drawdown_chart_js }}
                {{ returns_distribution_chart_js }}
                {{ monthly_returns_chart_js }}
            </script>
        </body>
        </html>
        """
    
    def generate_html_report(self, 
                            backtest_result: Dict[str, Any], 
                            metrics: Dict[str, Any],
                            output_file: Optional[str] = None) -> str:
        """
        Genera un reporte HTML con los resultados del backtest.
        
        Args:
            backtest_result: Resultados del backtest
            metrics: Métricas calculadas
            output_file: Nombre del archivo de salida (opcional)
            
        Returns:
            Ruta al archivo HTML generado
        """
        # Combinar resultados y métricas
        result_data = {**backtest_result, **metrics}
        
        # Generar gráficos
        equity_chart_js = self.generate_equity_curve_chart(backtest_result['equity_curve'])
        
        # Calcular drawdown si no está en el resultado
        if 'drawdown_curve' not in backtest_result:
            from metrics.metrics_calculator import MetricsCalculator
            dd_df, _ = MetricsCalculator.calculate_drawdown(backtest_result['equity_curve'])
            drawdown_chart_js = self.generate_drawdown_chart(dd_df)
        else:
            drawdown_chart_js = self.generate_drawdown_chart(backtest_result['drawdown_curve'])
        
        # Generar gráfico de distribución de rendimientos
        returns = backtest_result['result_data']['returns'].dropna()
        returns_distribution_chart_js = self.generate_returns_distribution_chart(returns)
        
        # Generar mapa de calor de rendimientos mensuales
        monthly_returns_chart_js = self.generate_monthly_returns_heatmap(returns)
        
        # Preparar datos de operaciones para la tabla
        trades_data = []
        for _, trade in backtest_result['trades'].iterrows():
            # Formatear duración
            if isinstance(trade['duration'], pd.Timedelta):
                duration_str = str(trade['duration']).split('.')[0]  # Eliminar microsegundos
            else:
                duration_str = "N/A"
            
            trades_data.append({
                'entry_time': trade['entry_time'].strftime('%Y-%m-%d %H:%M'),
                'entry_type': trade['entry_type'],
                'entry_price': float(trade['entry_price']),
                'exit_time': trade['exit_time'].strftime('%Y-%m-%d %H:%M') if pd.notna(trade['exit_time']) else "N/A",
                'exit_price': float(trade['exit_price']) if pd.notna(trade['exit_price']) else 0.0,
                'profit_loss': float(trade['profit_loss']) if pd.notna(trade['profit_loss']) else 0.0,
                'profit_loss_pct': float(trade['profit_loss_pct']) if pd.notna(trade['profit_loss_pct']) else 0.0,
                'duration_str': duration_str
            })
        
        # Preparar datos para la plantilla
        template_data = {
            'title': f"Reporte de Backtesting - {backtest_result['strategy_name']}",
            'subtitle': f"Período: {backtest_result['equity_curve'].index[0].strftime('%Y-%m-%d')} a {backtest_result['equity_curve'].index[-1].strftime('%Y-%m-%d')}",
            'metrics': metrics,
            'trades': trades_data,
            'strategy_name': backtest_result['strategy_name'],
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'equity_chart_js': equity_chart_js,
            'drawdown_chart_js': drawdown_chart_js,
            'returns_distribution_chart_js': returns_distribution_chart_js,
            'monthly_returns_chart_js': monthly_returns_chart_js,
            'isinstance': isinstance
        }
        

        # Cargar plantilla
        template_path = os.path.join(os.path.dirname(__file__), 'templates', 'report_template.html')
        if os.path.exists(template_path):
            with open(template_path, 'r') as f:
                template_str = f.read()
        else:
            template_str = self.default_template
        
        template = Template(template_str)
        
        # Generar HTML
        html_content = template.render(**template_data)
        
        # Guardar archivo
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"{backtest_result['strategy_name'].replace(' ', '_')}_{timestamp}.html"
        
        output_path = os.path.join(self.output_dir, output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path
    
    def generate_equity_curve_chart(self, equity_data: pd.DataFrame) -> str:
        """
        Genera el JavaScript para el gráfico de curva de equidad.
        
        Args:
            equity_data: DataFrame con datos de equidad
            
        Returns:
            JavaScript para crear el gráfico con Plotly
        """
        # Crear gráfico
        fig = go.Figure()
        
        # Añadir línea de equidad
        fig.add_trace(go.Scatter(
            x=equity_data.index,
            y=equity_data['equity'],
            mode='lines',
            name='Equidad',
            line=dict(color='rgb(49, 130, 189)', width=2)
        ))
        
        # Configurar diseño
        fig.update_layout(
            title='Curva de Equidad',
            xaxis_title='Fecha',
            yaxis_title='Equidad',
            hovermode='x unified',
            template='plotly_white'
        )
        
        # Convertir a JavaScript
        fig_json = fig.to_json()
        js_code = f"""
        var equityChartData = {fig_json};
        Plotly.newPlot('equity-chart', equityChartData.data, equityChartData.layout);
        """
        
        return js_code
    def generate_drawdown_chart(self, drawdown_data: pd.DataFrame) -> str:
        """
        Genera el JavaScript para el gráfico de drawdown.
        
        Args:
            drawdown_data: DataFrame con datos de drawdown
            
        Returns:
            JavaScript para crear el gráfico con Plotly
        """
        # Crear gráfico
        fig = go.Figure()
        
        # Añadir línea de drawdown
        fig.add_trace(go.Scatter(
            x=drawdown_data.index,
            y=-drawdown_data['drawdown_pct'],  # Negativo para mostrar hacia abajo
            mode='lines',
            name='Drawdown',
            fill='tozeroy',
            line=dict(color='rgba(255, 65, 54, 0.8)', width=1)
        ))
        
        # Configurar diseño
        fig.update_layout(
            title='Drawdown (%)',
            xaxis_title='Fecha',
            yaxis_title='Drawdown (%)',
            yaxis=dict(
                tickformat='.2f',
                ticksuffix='%',
                range=[None, 0]  # CAMBIAR ESTA LÍNEA: de rangemode='nonpositive' a range=[None, 0]
            ),
            hovermode='x unified',
            template='plotly_white'
        )
        # Convertir a JavaScript
        fig_json = fig.to_json()
        js_code = f"""
        var drawdownChartData = {fig_json};
        Plotly.newPlot('drawdown-chart', drawdownChartData.data, drawdownChartData.layout);
        """
        
        return js_code
    
    def generate_returns_distribution_chart(self, returns: pd.Series) -> str:
        """
        Genera el JavaScript para el gráfico de distribución de rendimientos.
        
        Args:
            returns: Serie con rendimientos diarios
            
        Returns:
            JavaScript para crear el gráfico con Plotly
        """
        # Convertir a porcentaje para mejor visualización
        returns_pct = returns * 100
        
        # Crear histograma
        fig = go.Figure()
        
        # Añadir histograma de rendimientos
        fig.add_trace(go.Histogram(
            x=returns_pct,
            name='Rendimientos',
            marker_color='rgba(49, 130, 189, 0.7)',
            xbins=dict(
                size=(returns_pct.max() - returns_pct.min()) / 50  # 50 bins
            )
        ))
        
        # Añadir línea vertical en 0
        fig.add_shape(
            type='line',
            x0=0, y0=0,
            x1=0, y1=1,
            yref='paper',
            line=dict(color='red', width=2, dash='dash')
        )
        
        # Configurar diseño
        fig.update_layout(
            title='Distribución de Rendimientos Diarios',
            xaxis_title='Rendimiento (%)',
            yaxis_title='Frecuencia',
            template='plotly_white',
            bargap=0.1
        )
        
        # Convertir a JavaScript
        fig_json = fig.to_json()
        js_code = f"""
        var returnsDistributionData = {fig_json};
        Plotly.newPlot('returns-distribution-chart', returnsDistributionData.data, returnsDistributionData.layout);
        """
        
        return js_code
    
    def generate_monthly_returns_heatmap(self, returns: pd.Series) -> str:
        """
        Genera el JavaScript para el mapa de calor de rendimientos mensuales.
        
        Args:
            returns: Serie con rendimientos diarios
            
        Returns:
            JavaScript para crear el gráfico con Plotly
        """
        # Asegurar que el índice es datetime
        if not isinstance(returns.index, pd.DatetimeIndex):
            # Intento de conversión, pero podría fallar dependiendo de los datos
            try:
                returns.index = pd.to_datetime(returns.index)
            except:
                # Si falla, devolver un gráfico vacío
                empty_js = """
                Plotly.newPlot('monthly-returns-chart', [], {title: 'No hay datos disponibles para rendimientos mensuales'});
                """
                return empty_js
        
        # Calcular rendimientos mensuales
        monthly_returns = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1) * 100
        
        # Crear DataFrame con Año y Mes como columnas
        monthly_df = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'return': monthly_returns.values
        })
        
        # Pivotar para crear matriz de año x mes
        pivot_table = monthly_df.pivot_table(
            index='year', 
            columns='month', 
            values='return',
            aggfunc='first'
        )
        
        # Nombres de meses para etiquetas
        month_names = [calendar.month_abbr[i] for i in range(1, 13)]
        
        # Crear mapa de calor
        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=month_names,
            y=pivot_table.index,
            colorscale='RdBu',
            zmid=0,  # Centrar en 0 para que rojo sea negativo y azul positivo
            text=np.around(pivot_table.values, 2),
            hovertemplate='%{y}, %{x}: %{z:.2f}%<extra></extra>',
            colorbar=dict(title='Rendimiento (%)')
        ))
        
        # Configurar diseño
        fig.update_layout(
            title='Rendimientos Mensuales (%)',
            xaxis_title='Mes',
            yaxis_title='Año',
            template='plotly_white'
        )
        
        # Convertir a JavaScript
        fig_json = fig.to_json()
        js_code = f"""
        var monthlyReturnsData = {fig_json};
        Plotly.newPlot('monthly-returns-chart', monthlyReturnsData.data, monthlyReturnsData.layout);
        """
        
        return js_code
    
    def generate_trades_table(self, trades: pd.DataFrame) -> str:
        """
        Genera el HTML para la tabla de operaciones.
        
        Args:
            trades: DataFrame con datos de operaciones
            
        Returns:
            HTML para la tabla de operaciones
        """
        # Convertir DataFrame a HTML
        html = """
        <div class="trades-table-container">
            <table id="trades-table">
                <thead>
                    <tr>
                        <th>Entrada</th>
                        <th>Tipo</th>
                        <th>Precio Entrada</th>
                        <th>Salida</th>
                        <th>Precio Salida</th>
                        <th>P/L</th>
                        <th>P/L %</th>
                        <th>Duración</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for _, trade in trades.iterrows():
            # Determinar clase CSS basada en el P/L
            row_class = "positive" if trade['profit_loss'] > 0 else "negative"
            
            # Formatear duración
            if isinstance(trade['duration'], pd.Timedelta):
                duration_str = str(trade['duration']).split('.')[0]  # Eliminar microsegundos
            else:
                duration_str = "N/A"
            
            html += f"""
                <tr class="{row_class}">
                    <td>{trade['entry_time'].strftime('%Y-%m-%d %H:%M')}</td>
                    <td>{trade['entry_type']}</td>
                    <td>{trade['entry_price']:.2f}</td>
                    <td>{trade['exit_time'].strftime('%Y-%m-%d %H:%M') if pd.notna(trade['exit_time']) else "N/A"}</td>
                    <td>{trade['exit_price']:.2f if pd.notna(trade['exit_price']) else 0.0}</td>
                    <td>{trade['profit_loss']:.2f if pd.notna(trade['profit_loss']) else 0.0}</td>
                    <td>{trade['profit_loss_pct']:.2f}% if pd.notna(trade['profit_loss_pct']) else 0.0</td>
                    <td>{duration_str}</td>
                </tr>
            """
        
        html += """
                </tbody>
            </table>
        </div>
        """
        
        return html