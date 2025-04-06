import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import calendar
import os
import io
import base64
from typing import Dict, Any, List, Optional, Tuple, Union
import warnings

# Suprimir advertencias de matplotlib
warnings.filterwarnings("ignore", category=UserWarning)


class MatplotlibReportPlotter:
    """
    Clase que genera visualizaciones para reportes utilizando matplotlib y seaborn
    """

    def __init__(self,
                 style: str = 'seaborn-v0_8-whitegrid',
                 fig_width: int = 10,
                 fig_height: int = 6,
                 dpi: int = 100):
        """
        Inicializa el generador de gráficos con matplotlib

        Args:
            style: Estilo de seaborn para los gráficos
            fig_width: Ancho de la figura en pulgadas
            fig_height: Alto de la figura en pulgadas
            dpi: Resolución de la figura en puntos por pulgada
        """
        self.style = style
        self.fig_width = fig_width
        self.fig_height = fig_height
        self.dpi = dpi
        self.setup_style()

    def setup_style(self):
        """Configura el estilo para todos los gráficos"""
        try:
            plt.style.use(self.style)
        except:
            plt.style.use('default')  # Fallback a estilo por defecto si falla

        # Configuraciones adicionales de estilo
        plt.rcParams['figure.figsize'] = (self.fig_width, self.fig_height)
        plt.rcParams['figure.dpi'] = self.dpi
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['font.family'] = 'sans-serif'

    def fig_to_base64(self, fig):
        """
        Convierte una figura de matplotlib a una cadena base64

        Args:
            fig: Figura de matplotlib

        Returns:
            Cadena base64 que representa la imagen
        """
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)
        return img_str

    def plot_equity_curve(self, equity_data: pd.DataFrame, initial_capital: float = 10000) -> str:
        """
        Genera un gráfico de la curva de equidad

        Args:
            equity_data: DataFrame con datos de equidad
            initial_capital: Capital inicial

        Returns:
            Imagen codificada en base64
        """
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))

        # Asegurar que equity_data.index es tipo datetime
        if not isinstance(equity_data.index, pd.DatetimeIndex):
            equity_data.index = pd.to_datetime(equity_data.index)

        # Asegurar que los datos de equidad son numéricos
        equity_values = pd.to_numeric(equity_data['equity'], errors='coerce')

        # Graficar curva de equidad
        ax.plot(equity_data.index, equity_values,
                linestyle='-',
                color='navy',
                linewidth=1.5,
                label='Capital')

        # Añadir línea de capital inicial
        ax.axhline(y=float(initial_capital),
                   linestyle='--',
                   color='red',
                   linewidth=1,
                   alpha=0.7,
                   label=f'Capital Inicial (${float(initial_capital):,.2f})')

        # Configurar el gráfico
        ax.set_title('Curva de Equidad', fontweight='bold')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Capital ($)')
        ax.grid(True, alpha=0.3)

        # Formato para fechas en el eje x
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)

        # Formato para valores en el eje y
        ax.yaxis.set_major_formatter(lambda x, pos: f'${x:,.0f}')

        # Añadir leyenda
        ax.legend(loc='best', frameon=True)

        # Ajustar diseño
        fig.tight_layout()

        # Convertir a base64
        return self.fig_to_base64(fig)

    def plot_drawdown(self, drawdown_data: pd.DataFrame, metrics: Dict[str, Any]) -> str:
        """
        Genera un gráfico de drawdown

        Args:
            drawdown_data: DataFrame con datos de drawdown
            metrics: Diccionario con métricas para mostrar en el gráfico

        Returns:
            Imagen codificada en base64
        """
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))

        # Asegurar que drawdown_data.index es tipo datetime
        if not isinstance(drawdown_data.index, pd.DatetimeIndex):
            drawdown_data.index = pd.to_datetime(drawdown_data.index)

        # Asegurar que los datos de drawdown son numéricos
        drawdown_values = pd.to_numeric(drawdown_data['drawdown_pct'], errors='coerce')

        # Graficar drawdown como área
        ax.fill_between(drawdown_data.index,
                        0,
                        -drawdown_values,
                        color='salmon',
                        alpha=0.5,
                        label='Drawdown')

        # Graficar drawdown como línea
        ax.plot(drawdown_data.index,
                -drawdown_values,
                color='red',
                linewidth=1,
                alpha=0.7)

        # Añadir línea de máximo drawdown
        max_dd = float(metrics.get('max_drawdown_pct', 0))
        ax.axhline(-max_dd,
                   linestyle='--',
                   color='darkred',
                   linewidth=1,
                   label=f'Máximo Drawdown: {max_dd:.2f}%')

        # Configurar el gráfico
        ax.set_title('Drawdown', fontweight='bold')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)

        # Formato para fechas en el eje x
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)

        # Invertir el eje y para que el drawdown se muestre hacia abajo
        ax.invert_yaxis()

        # Añadir leyenda
        ax.legend(loc='lower left', frameon=True)

        # Añadir métricas como texto
        info_text = (
            f"Capital Inicial: ${float(metrics.get('initial_capital', 0)):,.2f}\n"
            f"Capital Final: ${float(metrics.get('final_equity', 0)):,.2f}\n"
            f"Retorno Total: {float(metrics.get('total_return_pct', 0)):.2f}%\n"
            f"CAGR: {float(metrics.get('cagr_pct', 0)):.2f}%\n"
            f"Volatilidad Anualizada: {float(metrics.get('volatility_pct', 0)):.2f}%\n"
            f"Máximo Drawdown: {max_dd:.2f}%\n"
            f"Ratio de Calmar: {float(metrics.get('calmar_ratio', 0)):.2f}\n"
            f"Total Operaciones: {int(metrics.get('total_trades', 0))}\n"
            f"Win Rate: {float(metrics.get('win_rate', 0)):.2f}%\n"
        )

        # Colocar texto en la esquina inferior derecha
        ax.text(0.02, 0.05, info_text,
                transform=ax.transAxes,
                verticalalignment='bottom',
                horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # Ajustar diseño
        fig.tight_layout()

        # Convertir a base64
        return self.fig_to_base64(fig)

    def plot_returns_distribution(self, returns: pd.Series) -> Tuple[str, Dict[str, Any]]:
        """
        Genera un gráfico de distribución de rendimientos

        Args:
            returns: Serie con rendimientos diarios

        Returns:
            Tuple con (imagen codificada en base64, estadísticas de retornos)
        """
        # Convertir a porcentaje y asegurar valores numéricos
        returns_pct = pd.to_numeric(returns, errors='coerce') * 100

        # Calcular estadísticas
        daily_stats = {
            'mean': float(returns_pct.mean()),
            'median': float(returns_pct.median()),
            'std': float(returns_pct.std()),
            'min': float(returns_pct.min()),
            'max': float(returns_pct.max()),
            'skew': float(returns_pct.skew()),
            'kurtosis': float(returns_pct.kurt())
        }

        # Calcular estadísticas mensuales (usar 'ME' en lugar de 'M')
        monthly_returns = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1) * 100
        monthly_returns = pd.to_numeric(monthly_returns, errors='coerce')

        monthly_stats = {
            'mean': float(monthly_returns.mean()),
            'median': float(monthly_returns.median()),
            'std': float(monthly_returns.std()),
            'min': float(monthly_returns.min()),
            'max': float(monthly_returns.max()),
            'skew': float(monthly_returns.skew()),
            'kurtosis': float(monthly_returns.kurt())
        }

        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))

        # Histograma con KDE
        sns.histplot(returns_pct, bins=50, kde=True, color='royalblue', ax=ax, stat='density', alpha=0.6)

        # Añadir línea vertical en 0
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.7)

        # Marcar la media y la mediana
        ax.axvline(x=daily_stats['mean'], color='green', linestyle='-', linewidth=1.5, alpha=0.7,
                   label=f'Media: {daily_stats["mean"]:.2f}%')
        ax.axvline(x=daily_stats['median'], color='purple', linestyle=':', linewidth=1.5, alpha=0.7,
                   label=f'Mediana: {daily_stats["median"]:.2f}%')

        # Configurar el gráfico
        ax.set_title('Distribución de Rendimientos Diarios', fontweight='bold')
        ax.set_xlabel('Rendimiento Diario (%)')
        ax.set_ylabel('Densidad')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')

        # Ajustar diseño
        fig.tight_layout()

        # Combinar estadísticas
        returns_stats = {
            'daily': daily_stats,
            'monthly': monthly_stats
        }

        # Convertir a base64
        return self.fig_to_base64(fig), returns_stats

    def plot_monthly_returns_heatmap(self, returns: pd.Series) -> Tuple[str, Dict[int, Dict[int, float]]]:
        """
        Genera un mapa de calor de rendimientos mensuales

        Args:
            returns: Serie con rendimientos diarios

        Returns:
            Tuple con (imagen codificada en base64, tabla de rendimientos mensuales)
        """
        # Calcular rendimientos mensuales (usar 'ME' en lugar de 'M')
        monthly_returns = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1) * 100
        monthly_returns = pd.to_numeric(monthly_returns, errors='coerce')

        # Crear DataFrame con Año y Mes como columnas
        monthly_df = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'return': monthly_returns.values
        })

        # Pivotar para crear matriz de año x mes
        try:
            pivot = monthly_df.pivot_table(
                index='year',
                columns='month',
                values='return',
                aggfunc='first'
            )

            # Rellenar NaNs con 0s para visualización
            pivot_filled = pivot.fillna(0)

            # Calcular rendimientos anuales
            yearly_returns = {}
            for year in pivot.index:
                yearly_returns[year] = float(pivot.loc[year].sum())

            # Crear diccionario para la tabla de rendimientos mensuales
            monthly_returns_table = {}
            for year in pivot.index:
                monthly_returns_table[year] = {}
                for month in range(1, 13):
                    if month in pivot.columns and not pd.isna(pivot.loc[year, month]):
                        monthly_returns_table[year][month] = float(pivot.loc[year, month])
                monthly_returns_table[year]['year_total'] = yearly_returns[year]

            # Crear gráfico
            fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))

            # Crear mapa de calor
            cmap = sns.diverging_palette(10, 240, as_cmap=True)
            heatmap = sns.heatmap(
                pivot_filled,
                cmap=cmap,
                center=0,
                annot=True,
                fmt='.1f',
                linewidths=0.5,
                ax=ax,
                cbar_kws={'label': 'Rendimiento (%)'}
            )

            # Configurar etiquetas de los ejes
            month_names = [calendar.month_abbr[i] for i in range(1, 13)]
            ax.set_xticklabels(month_names, rotation=0)
            ax.set_yticklabels(pivot.index, rotation=0)

            # Configurar el gráfico
            ax.set_title('Rendimientos Mensuales (%)', fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel('Año')

            # Añadir columna de rendimiento anual
            ax_twin = ax.twinx()
            for i, year in enumerate(pivot.index):
                ax_twin.text(
                    12.5, i + 0.5,
                    f"{yearly_returns[year]:.1f}%",
                    ha='left', va='center',
                    fontweight='bold',
                    color='green' if yearly_returns[year] > 0 else 'red'
                )
            ax_twin.set_ylim(ax.get_ylim())
            ax_twin.set_yticks([])

            # Ajustar diseño
            fig.tight_layout()

            # Convertir a base64
            return self.fig_to_base64(fig), monthly_returns_table

        except Exception as e:
            # Si hay un error, crear un gráfico vacío
            fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
            ax.text(0.5, 0.5, f"No hay suficientes datos para mostrar rendimientos mensuales\n{str(e)}",
                    ha='center', va='center', fontsize=12)
            ax.axis('off')

            # Convertir a base64
            return self.fig_to_base64(fig), {}

    def extract_drawdown_periods(self, drawdown_data: pd.DataFrame, threshold: float = 5.0) -> List[Dict[str, Any]]:
        """
        Extrae los períodos de drawdown significativos

        Args:
            drawdown_data: DataFrame con datos de drawdown
            threshold: Umbral de drawdown para considerar un período como significativo (%)

        Returns:
            Lista de períodos de drawdown
        """
        drawdown_periods = []
        in_drawdown = False
        start_idx = None
        max_dd = 0

        # Asegurar que drawdown_data['drawdown_pct'] es numérico
        drawdown_pct = pd.to_numeric(drawdown_data['drawdown_pct'], errors='coerce')

        # Recorrer el DataFrame
        for idx, dd_pct in zip(drawdown_data.index, drawdown_pct):
            # Inicio de un nuevo período de drawdown
            if not in_drawdown and dd_pct >= threshold:
                in_drawdown = True
                start_idx = idx
                max_dd = dd_pct

            # Actualizar máximo drawdown en un período existente
            elif in_drawdown and dd_pct > max_dd:
                max_dd = dd_pct

            # Fin de un período de drawdown
            elif in_drawdown and dd_pct < threshold:
                in_drawdown = False

                # Añadir el período
                drawdown_periods.append({
                    'start_date': start_idx.strftime('%Y-%m-%d'),
                    'end_date': idx.strftime('%Y-%m-%d'),
                    'duration': (idx - start_idx).days,
                    'depth': float(max_dd),
                    'recovery': 'Completa'
                })

                # Resetear variables
                start_idx = None
                max_dd = 0

        # Si hay un período activo al final
        if in_drawdown:
            drawdown_periods.append({
                'start_date': start_idx.strftime('%Y-%m-%d'),
                'end_date': drawdown_data.index[-1].strftime('%Y-%m-%d'),
                'duration': (drawdown_data.index[-1] - start_idx).days,
                'depth': float(max_dd),
                'recovery': 'En progreso'
            })

        # Ordenar por profundidad
        drawdown_periods.sort(key=lambda x: x['depth'], reverse=True)

        return drawdown_periods[:5]  # Retorna los 5 períodos más profundos