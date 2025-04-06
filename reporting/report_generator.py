import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import base64
import io
import logging
import re
from typing import Dict, Any, List, Optional, Tuple, Union
from jinja2 import Template
import json
import plotly.graph_objects as go
import plotly.utils

# Importar las clases auxiliares
try:
    from reporting.matplotlib_report_plotter import MatplotlibReportPlotter
except ImportError:
    # Para uso local, si el módulo no está en reporting/
    try:
        from matplotlib_report_plotter import MatplotlibReportPlotter
    except ImportError:
        print("ERROR: No se pudo importar MatplotlibReportPlotter")

class ReportGenerator:
    """
    Generador de reportes HTML para backtesting de estrategias.
    """

    def __init__(self,
                template_path: Optional[str] = None,
                output_dir: str = "./reports",
                engine: str = "plotly",
                template_name: Optional[str] = None):
        """
        Inicializa el generador de reportes.
        """
        self.template_path = template_path
        self.output_dir = output_dir
        self.engine = engine.lower()
        self.template_name = template_name

        # Validar motor de gráficos
        if self.engine not in ["plotly", "matplotlib"]:
            logging.warning(f"Motor de gráficos no válido: {engine}. Usando 'plotly' por defecto.")
            self.engine = "plotly"

        # Crear directorio si no existe
        os.makedirs(output_dir, exist_ok=True)

        # Inicializar generador de gráficos matplotlib si es necesario
        if self.engine == "matplotlib":
            self.plotter = MatplotlibReportPlotter()

        # Configurar directorio de plantillas
        self.templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
        if not os.path.exists(self.templates_dir):
            logging.warning(f"Directorio de plantillas no encontrado: {self.templates_dir}")

        # Plantilla HTML por defecto
        self.default_template = """
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="UTF-8">
            <title>Reporte de Backtesting</title>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; }
                .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
                h1 { color: #333; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{{ title }}</h1>
                <p>{{ subtitle }}</p>
                <p>No se pudo generar el reporte completo debido a un error.</p>
                <p>Estrategia: {{ strategy_name }}</p>
                <p>Generado el: {{ generation_time }}</p>
            </div>
        </body>
        </html>
        """

    def generate_html_report(self,
                            backtest_result: Dict[str, Any],
                            metrics: Dict[str, Any],
                            output_file: Optional[str] = None) -> str:
        """
        Genera un reporte HTML con los resultados del backtest.
        """
        # Seleccionar motor de generación de gráficos
        if self.engine == "matplotlib":
            return self._generate_matplotlib_report(backtest_result, metrics, output_file)
        else:
            return self._generate_plotly_report(backtest_result, metrics, output_file)

    def _generate_matplotlib_report(self,
                                  backtest_result: Dict[str, Any],
                                  metrics: Dict[str, Any],
                                  output_file: Optional[str] = None) -> str:
        """
        Genera un reporte HTML con gráficos de Matplotlib.
        """
        try:
            # Verificar que tenemos el plotter inicializado
            if not hasattr(self, 'plotter'):
                self.plotter = MatplotlibReportPlotter()

            # Obtener datos necesarios
            equity_curve = backtest_result['equity_curve']

            # Calcular drawdown si no está en el resultado
            if 'drawdown_curve' not in backtest_result:
                from metrics.metrics_calculator import MetricsCalculator
                drawdown_df, _ = MetricsCalculator.calculate_drawdown(equity_curve)
            else:
                drawdown_df = backtest_result['drawdown_curve']

            # Obtener retornos
            returns = backtest_result['result_data']['returns'].dropna()

            # Generar gráficos
            equity_curve_plot = self.plotter.plot_equity_curve(
                equity_curve,
                initial_capital=backtest_result.get('initial_capital', 10000)
            )

            drawdown_plot = self.plotter.plot_drawdown(drawdown_df, metrics)

            returns_distribution_plot, returns_stats = self.plotter.plot_returns_distribution(returns)

            monthly_returns_plot, monthly_returns_table = self.plotter.plot_monthly_returns_heatmap(returns)

            # Extraer períodos de drawdown significativos
            drawdown_periods = self.plotter.extract_drawdown_periods(drawdown_df)

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
                    'exit_price': float(trade['exit_price']),
                    'profit_loss': float(trade['profit_loss']) if pd.notna(trade['profit_loss']) else 0.0,
                    'profit_loss_pct': float(trade['profit_loss_pct']) if pd.notna(trade['profit_loss_pct']) else 0.0,
                    'duration_str': duration_str
                })

            # Añadir métricas adicionales necesarias para la plantilla profesional
            if 'win_rate' in metrics:
                metrics['loss_rate'] = 100 - metrics['win_rate']

            # Calcular duración media de operaciones en formato legible
            if 'avg_duration_days' in metrics:
                avg_days = metrics['avg_duration_days']
                days = int(avg_days)
                hours = int((avg_days - days) * 24)
                metrics['avg_trade_duration_str'] = f"{days} días, {hours} horas"

            # Añadir volatilidad anualizada si no existe
            if 'volatility_pct' in metrics:
                metrics['annual_volatility_pct'] = metrics['volatility_pct']

            # Preparar datos para la plantilla
            template_data = {
                'title': f"Reporte de Backtesting - {backtest_result['strategy_name']}",
                'subtitle': f"Período: {equity_curve.index[0].strftime('%Y-%m-%d')} a {equity_curve.index[-1].strftime('%Y-%m-%d')}",
                'metrics': metrics,
                'trades': trades_data,
                'strategy_name': backtest_result['strategy_name'],
                'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'equity_curve_plot': equity_curve_plot,
                'drawdown_plot': drawdown_plot,
                'returns_distribution_plot': returns_distribution_plot,
                'monthly_returns_plot': monthly_returns_plot,
                'returns_stats': returns_stats,
                'monthly_returns_table': monthly_returns_table,
                'drawdown_periods': drawdown_periods,
                'current_year': datetime.now().strftime('%Y')  # Año actual para reemplazar {% now 'Y' %}
            }

            # Cargar plantilla simplificada
            template_str = self._load_template()

            # Generar HTML
            template = Template(template_str)
            html_content = template.render(**template_data)

            # Guardar archivo
            if output_file is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = f"{backtest_result['strategy_name'].replace(' ', '_')}_{timestamp}.html"

            output_path = os.path.join(self.output_dir, output_file)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            return output_path

        except Exception as e:
            logging.error(f"Error al generar el reporte: {e}")

            # Generar un reporte básico con la plantilla por defecto
            template_data = {
                'title': f"Reporte Básico - {backtest_result['strategy_name']}",
                'subtitle': f"Se produjo un error al generar el reporte completo",
                'strategy_name': backtest_result['strategy_name'],
                'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            template = Template(self.default_template)
            html_content = template.render(**template_data)

            # Guardar archivo de error
            if output_file is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = f"{backtest_result['strategy_name'].replace(' ', '_')}_{timestamp}_error.html"

            output_path = os.path.join(self.output_dir, output_file)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            # Re-lanzar la excepción para que el llamador sepa que hubo un error
            raise

    def _get_template_filename(self) -> str:
        """
        Obtiene la ruta completa al archivo de plantilla a utilizar.
        """
        # Si se especificó un nombre de plantilla
        if self.template_name:
            # Comprobar si es 'professional' y usar el nombre correcto del archivo
            if self.template_name.lower() == 'professional':
                template_file = os.path.join(self.templates_dir, "professional_template.html")
                if os.path.exists(template_file):
                    return template_file
                else:
                    logging.warning(f"Plantilla professional_template.html no encontrada")
            else:
                # Probar con el nombre exacto
                template_file = os.path.join(self.templates_dir, f"{self.template_name}.html")
                if os.path.exists(template_file):
                    return template_file

                # Probar con nombre + _template
                template_file = os.path.join(self.templates_dir, f"{self.template_name}_template.html")
                if os.path.exists(template_file):
                    return template_file

                logging.warning(f"Plantilla {self.template_name}.html no encontrada")

        # Si se especificó un path de plantilla
        if self.template_path and os.path.exists(self.template_path):
            return self.template_path

        # Intentar cargar la plantilla adecuada según el motor
        if self.engine == "matplotlib":
            template_file = os.path.join(self.templates_dir, "professional_template.html")
            if os.path.exists(template_file):
                return template_file
        else:
            template_file = os.path.join(self.templates_dir, "report_template.html")
            if os.path.exists(template_file):
                return template_file

        # Si todo falla, usar la plantilla por defecto
        return "default_template"

    def _load_template(self) -> str:
        """
        Carga la plantilla HTML a utilizar y realiza modificaciones necesarias.
        """
        template_file = self._get_template_filename()

        if template_file == "default_template":
            return self.default_template

        try:
            with open(template_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Solución: Reemplazar la etiqueta {% now 'Y' %} por {{ current_year }}
            content = re.sub(r"{%\s*now\s+['\"]\w+['\"]?\s*%}", "{{ current_year }}", content)

            return content
        except Exception as e:
            logging.error(f"Error al cargar la plantilla: {e}")
            return self.default_template
    def _generate_plotly_report(self,
                                backtest_result: Dict[str, Any],
                                metrics: Dict[str, Any],
                                output_file: Optional[str] = None) -> str:
        """
        Genera un reporte HTML con gráficos de Plotly.

        Args:
            backtest_result: Resultados del backtest
            metrics: Métricas calculadas
            output_file: Nombre del archivo de salida (opcional)

        Returns:
            Ruta al archivo HTML generado
        """
        # Este código es básicamente el contenido actual del método generate_html_report
        # Simplemente se mueve a este método auxiliar

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
                'exit_price': float(trade['exit_price']),
                'profit_loss': float(trade['profit_loss']) if pd.notna(trade['profit_loss']) else 0.0,
                'profit_loss_pct': float(trade['profit_loss_pct']) if pd.notna(trade['profit_loss_pct']) else 0.0,
                'duration_str': duration_str
            })

        # Pre-procesar las métricas para garantizar que todos los valores sean simples (no diccionarios)
        processed_metrics = {}
        for key, value in metrics.items():
            if key not in ['equity_curve', 'trades', 'result_data']:
                # Si es un diccionario, convertirlo a string
                if isinstance(value, dict):
                    processed_metrics[key] = str(value)
                # Si es un número, mantenerlo como número
                elif isinstance(value, (int, float)):
                    processed_metrics[key] = value
                # Para otros tipos, convertir a string
                else:
                    processed_metrics[key] = str(value)
            else:
                # Mantener los objetos de datos grandes sin cambios
                processed_metrics[key] = value

        trades_table_html = self._generate_trades_table(backtest_result['trades'])

        # Preparar datos para la plantilla
        template_data = {
            'title': f"Reporte de Backtesting - {backtest_result['strategy_name']}",
            'subtitle': f"Período: {backtest_result['equity_curve'].index[0].strftime('%Y-%m-%d')} a {backtest_result['equity_curve'].index[-1].strftime('%Y-%m-%d')}",
            'metrics': processed_metrics,
            'trades': trades_data,
            'strategy_name': backtest_result['strategy_name'],
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'equity_chart_js': equity_chart_js,
            'drawdown_chart_js': drawdown_chart_js,
            'returns_distribution_chart_js': returns_distribution_chart_js,
            'monthly_returns_chart_js': monthly_returns_chart_js,
            'trades_table_html': trades_table_html,
            'isinstance': isinstance,
            'now': lambda fmt: datetime.now().strftime(fmt)  # Función now para la plantilla
        }

        # Cargar plantilla
        template_str = self._load_template()
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


        # Convertir los datos a listas puras de Python
        timestamps = equity_data.index.astype(str).tolist()  # Convertir fechas a strings
        equities = equity_data['equity'].astype(float).tolist()  # Asegurar floats y convertir a lista


        # Crear la figura
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=timestamps,  # Lista de strings
            y=equities,  # Lista de floats
            mode='lines',
            name='Equidad',
            line=dict(color='rgb(49, 130, 189)', width=2)
        ))

        # Serializar correctamente usando el encoder de Plotly
        fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

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
        # Convertir drawdown_pct a lista de floats
        drawdown_pct = drawdown_data['drawdown_pct'].astype(float).tolist()  # Convertir a lista de floats
        timestamps = drawdown_data.index.astype(str).tolist()  # Convertir fechas a string

        # Aplicar signo negativo a cada elemento de drawdown_pct
        drawdown_pct = [-x for x in drawdown_pct]  # Invertir el signo de cada valor de drawdown_pct

        print(type(timestamps), type(drawdown_pct))  # Verifica que sean listas
        print(timestamps[:5], drawdown_pct[:5])  # Imprime una muestra para ver la conversión

        # Crear el gráfico
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=timestamps,  # Lista de fechas como strings
            y=drawdown_pct,  # Lista de drawdowns negativos
            mode='lines',
            name='Drawdown',
            fill='tozeroy',
            line=dict(color='rgba(255, 65, 54, 0.8)', width=1)
        ))

        # Configurar layout
        fig.update_layout(
            yaxis=dict(
                tickformat='.2f',
                ticksuffix='%',
                range=[min(drawdown_pct) * 1.1, 0],  # Ajuste para mostrar el rango completo
                title='Drawdown (%)'
            ),
            title='Drawdown (%)',
            xaxis_title='Fecha',
            yaxis_title='Drawdown (%)',
            hovermode='x unified',
            template='plotly_white'
        )

        # Convertir a JavaScript
        fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

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
        # Convertir a porcentaje para mejor visualización y a lista Python
        returns_pct = (returns * 100).astype(float).tolist()

        # Calcular tamaño de bins de forma robusta
        if len(returns_pct) > 1:
            bin_size = (max(returns_pct) - min(returns_pct)) / 50
        else:
            bin_size = 0.1

        # Crear histograma
        fig = go.Figure()

        # Añadir histograma de rendimientos
        fig.add_trace(go.Histogram(
            x=returns_pct,
            name='Rendimientos',
            marker_color='rgba(49, 130, 189, 0.7)',
            xbins=dict(size=bin_size)
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

        # Convertir a JavaScript usando el encoder correcto
        fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

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
        try:
            # Asegurar que el índice es datetime
            if not isinstance(returns.index, pd.DatetimeIndex):
                try:
                    returns.index = pd.to_datetime(returns.index)
                except:
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

            # Preparar datos para Plotly
            z_values = pivot_table.values.tolist()  # Convertir a lista de listas Python
            y_values = pivot_table.index.astype(int).tolist()  # Años como enteros

            # Nombres de meses para etiquetas
            month_names = [calendar.month_abbr[i] for i in range(1, 13)]

            # Asegurar que tenemos valores para el texto (redondeados)
            text_values = np.round(pivot_table.values, 2).tolist()

            # Crear mapa de calor
            fig = go.Figure(data=go.Heatmap(
                z=z_values,
                x=month_names,
                y=y_values,
                colorscale='RdBu',
                zmid=0,  # Centrar en 0 para que rojo sea negativo y azul positivo
                text=text_values,
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

            # Convertir a JavaScript usando el encoder correcto
            fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

            js_code = f"""
            var monthlyReturnsData = {fig_json};
            Plotly.newPlot('monthly-returns-chart', monthlyReturnsData.data, monthlyReturnsData.layout);
            """

            return js_code

        except Exception as e:
            print(f"Error en generate_monthly_returns_heatmap: {e}")
            empty_js = f"""
            Plotly.newPlot('monthly-returns-chart', [], {{title: 'Error al generar el mapa de calor: {str(e)}'}});
            """
            return empty_js

    def _generate_trades_table(self, trades: pd.DataFrame) -> str:
        """
        Genera el HTML para la tabla de operaciones con información detallada.

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
                        <th>Tamaño Posición Est.</th>
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

            # Calcular tamaño de posición estimado basado en P/L
            if trade['entry_type'] == 'BUY':
                price_diff = float(trade['exit_price']) - float(trade['entry_price'])
            else:
                price_diff = float(trade['entry_price']) - float(trade['exit_price'])

            # Evitar división por cero
            estimated_position_size = float('NaN')
            if abs(price_diff) > 0.0001:
                estimated_position_size = float(trade['profit_loss']) / price_diff

            html += f"""
                <tr class="{row_class}">
                    <td>{trade['entry_time'].strftime('%Y-%m-%d %H:%M') if hasattr(trade['entry_time'], 'strftime') else trade['entry_time']}</td>
                    <td>{trade['entry_type']}</td>
                    <td>{float(trade['entry_price']):.2f}</td>
                    <td>{trade['exit_time'].strftime('%Y-%m-%d %H:%M') if hasattr(trade['exit_time'], 'strftime') and pd.notna(trade['exit_time']) else trade['exit_time'] if pd.notna(trade['exit_time']) else "N/A"}</td>
                    <td>{float(trade['exit_price'])}</td>
                    <td>{float(trade['profit_loss']):.2f}</td>
                    <td>{float(trade['profit_loss_pct']):.2f}%</td>
                    <td>{duration_str}</td>
                    <td>{estimated_position_size:.4f}</td>
                </tr>
            """

        html += """
                </tbody>
            </table>
        </div>
        <div class="trade-explanation" style="margin-top: 10px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
            <p><strong>Nota sobre el P/L:</strong> El P/L mostrado es un múltiplo de la diferencia de precio porque se están operando múltiples unidades.</p>
            <p><strong>Fórmula:</strong> P/L = (Precio Salida - Precio Entrada) × Tamaño Posición (para operaciones largas)</p>
            <p>La columna "Tamaño Posición Est." muestra cuántas unidades se operaron en cada trade, calculado como P/L ÷ DiferenciaPrecio.</p>
        </div>
        """

        return html