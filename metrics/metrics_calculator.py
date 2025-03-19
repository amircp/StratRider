import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import math

class MetricsCalculator:
    """
    Calculador de métricas para evaluar el rendimiento de estrategias de trading.
    """
    
    @staticmethod
    def calculate_drawdown(equity_curve: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Calcula el drawdown a partir de la curva de equidad.
        
        Args:
            equity_curve: DataFrame con la curva de equidad
            
        Returns:
            DataFrame con el drawdown y diccionario con métricas de drawdown
        """
        # Asegurar que equity_curve tiene la columna correcta
        if 'equity' not in equity_curve.columns:
            raise ValueError("La curva de equidad debe tener una columna 'equity'")
        
        # Calcular drawdown
        equity = equity_curve['equity']
        
        # Calcular el máximo acumulado
        rolling_max = equity.cummax()
        
        # Calcular drawdown en valores absolutos y porcentuales
        drawdown = rolling_max - equity
        drawdown_pct = (drawdown / rolling_max) * 100
        
        # Crear DataFrame con los resultados
        dd_df = pd.DataFrame({
            'equity': equity,
            'peak': rolling_max,
            'drawdown': drawdown,
            'drawdown_pct': drawdown_pct
        })
        
        # Calcular métricas de drawdown
        max_drawdown = drawdown_pct.max()
        
        # Calcular períodos de drawdown
        is_recovery = drawdown_pct == 0
        is_drawdown_start = is_recovery.shift(1).fillna(True) & ~is_recovery
        dd_periods = []
        
        current_dd_start = None
        current_peak = None
        
        for idx, row in dd_df.iterrows():
            if row['drawdown_pct'] == 0:
                if current_dd_start is not None:
                    # Fin de un período de drawdown
                    dd_periods.append({
                        'start': current_dd_start,
                        'end': idx,
                        'duration': (idx - current_dd_start).days,
                        'max_dd_pct': dd_df.loc[current_dd_start:idx, 'drawdown_pct'].max()
                    })
                    current_dd_start = None
                    current_peak = None
            else:
                if current_dd_start is None:
                    # Inicio de un nuevo período de drawdown
                    current_dd_start = idx
                    current_peak = row['peak']
        
        # Si hay un drawdown activo al final del período
        if current_dd_start is not None:
            dd_periods.append({
                'start': current_dd_start,
                'end': dd_df.index[-1],
                'duration': (dd_df.index[-1] - current_dd_start).days,
                'max_dd_pct': dd_df.loc[current_dd_start:, 'drawdown_pct'].max()
            })
        
        # Calcular estadísticas de drawdown
        avg_dd_duration = np.mean([period['duration'] for period in dd_periods]) if dd_periods else 0
        max_dd_duration = max([period['duration'] for period in dd_periods]) if dd_periods else 0
        
        # Crear diccionario con métricas
        dd_metrics = {
            'max_drawdown_pct': max_drawdown,
            'avg_drawdown_duration': avg_dd_duration,
            'max_drawdown_duration': max_dd_duration,
            'drawdown_periods': len(dd_periods)
        }
        
        return dd_df, dd_metrics
    
    @staticmethod
    def calculate_win_rate(trades: pd.DataFrame) -> float:
        """
        Calcula la tasa de operaciones ganadoras.
        
        Args:
            trades: DataFrame con operaciones
            
        Returns:
            Tasa de operaciones ganadoras (0-1)
        """
        if trades.empty:
            return 0.0
        
        winning_trades = trades[trades['profit_loss'] > 0]
        return len(winning_trades) / len(trades)
    
    @staticmethod
    def calculate_profit_factor(trades: pd.DataFrame) -> float:
        """
        Calcula el factor de beneficio (total de ganancias / total de pérdidas).
        
        Args:
            trades: DataFrame con operaciones
            
        Returns:
            Factor de beneficio
        """
        if trades.empty:
            return 0.0
        
        winning_trades = trades[trades['profit_loss'] > 0]
        losing_trades = trades[trades['profit_loss'] < 0]
        
        total_profit = winning_trades['profit_loss'].sum() if not winning_trades.empty else 0
        total_loss = abs(losing_trades['profit_loss'].sum()) if not losing_trades.empty else 0
        
        if total_loss == 0:
            return float('inf') if total_profit > 0 else 0.0
        
        return total_profit / total_loss
    
    @staticmethod
    def calculate_expectancy(trades: pd.DataFrame) -> Dict[str, float]:
        """
        Calcula la expectativa matemática por operación.
        
        Args:
            trades: DataFrame con operaciones
            
        Returns:
            Diccionario con métricas de expectativa
        """
        if trades.empty:
            return {
                'expectancy': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'win_rate': 0.0,
                'r_expectancy': 0.0
            }
        
        winning_trades = trades[trades['profit_loss'] > 0]
        losing_trades = trades[trades['profit_loss'] < 0]
        
        win_rate = len(winning_trades) / len(trades) if not trades.empty else 0
        avg_win = winning_trades['profit_loss'].mean() if not winning_trades.empty else 0
        avg_loss = losing_trades['profit_loss'].mean() if not losing_trades.empty else 0
        
        # Expectancy in currency
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        # R-expectancy (reward-to-risk ratio)
        r_expectancy = 0.0
        if avg_loss != 0:
            r_expectancy = win_rate * (abs(avg_win) / abs(avg_loss)) - (1 - win_rate)
        
        return {
            'expectancy': expectancy,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_rate': win_rate,
            'r_expectancy': r_expectancy
        }
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """
        Calcula el ratio de Sharpe.
        
        Args:
            returns: Serie con retornos diarios
            risk_free_rate: Tasa libre de riesgo anualizada
            periods_per_year: Número de períodos por año
            
        Returns:
            Ratio de Sharpe anualizado
        """
        if returns.empty:
            return 0.0
        
        # Convertir tasa libre de riesgo a diaria
        daily_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
        
        # Calcular exceso de retorno
        excess_returns = returns - daily_rf
        
        # Calcular Sharpe
        sharpe_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0
        
        # Anualizar
        sharpe_ratio *= np.sqrt(periods_per_year)
        
        return sharpe_ratio
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """
        Calcula el ratio de Sortino.
        
        Args:
            returns: Serie con retornos diarios
            risk_free_rate: Tasa libre de riesgo anualizada
            periods_per_year: Número de períodos por año
            
        Returns:
            Ratio de Sortino anualizado
        """
        if returns.empty:
            return 0.0
        
        # Convertir tasa libre de riesgo a diaria
        daily_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
        
        # Calcular exceso de retorno
        excess_returns = returns - daily_rf
        
        # Calcular desviación de retornos negativos
        negative_returns = excess_returns[excess_returns < 0]
        downside_deviation = negative_returns.std() if not negative_returns.empty else 0
        
        # Calcular Sortino
        sortino_ratio = excess_returns.mean() / downside_deviation if downside_deviation != 0 else 0
        
        # Anualizar
        sortino_ratio *= np.sqrt(periods_per_year)
        
        return sortino_ratio
    
    @staticmethod
    def calculate_cagr(equity_curve: pd.DataFrame, days: Optional[int] = None) -> float:
        """
        Calcula la tasa de crecimiento anual compuesto.
        
        Args:
            equity_curve: DataFrame con la curva de equidad
            days: Número de días del período (opcional, se calcula automáticamente)
            
        Returns:
            CAGR como porcentaje
        """
        if equity_curve.empty:
            return 0.0
        
        if 'equity' not in equity_curve.columns:
            raise ValueError("La curva de equidad debe tener una columna 'equity'")
        
        # Obtener valores inicial y final
        initial_value = equity_curve['equity'].iloc[0]
        final_value = equity_curve['equity'].iloc[-1]
        
        # Calcular número de días si no se proporciona
        if days is None:
            days = (equity_curve.index[-1] - equity_curve.index[0]).days
            if days == 0:
                days = 1  # Evitar división por cero
        
        # Calcular CAGR
        years = days / 365.25
        cagr = (final_value / initial_value) ** (1 / years) - 1
        
        return cagr * 100  # Convertir a porcentaje
    
    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series, max_drawdown: float, periods_per_year: int = 252) -> float:
        """
        Calcula el ratio de Calmar.
        
        Args:
            returns: Serie con retornos diarios
            max_drawdown: Máximo drawdown en porcentaje
            periods_per_year: Número de períodos por año
            
        Returns:
            Ratio de Calmar
        """
        if returns.empty or max_drawdown == 0:
            return 0.0
        
        # Calcular retorno anualizado
        annualized_return = returns.mean() * periods_per_year * 100
        
        # Calcular Calmar
        calmar_ratio = annualized_return / max_drawdown
        
        return calmar_ratio
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calcula el Valor en Riesgo (VaR) histórico.
        
        Args:
            returns: Serie con retornos diarios
            confidence_level: Nivel de confianza (0-1)
            
        Returns:
            VaR como porcentaje positivo
        """
        if returns.empty:
            return 0.0
        
        # Calcular percentil
        var = np.percentile(returns, 100 * (1 - confidence_level))
        
        return abs(var) * 100  # Convertir a porcentaje positivo
    
    @staticmethod
    def calculate_max_consecutive_wins_losses(trades: pd.DataFrame) -> Dict[str, int]:
        """
        Calcula el máximo número de operaciones consecutivas ganadoras y perdedoras.
        
        Args:
            trades: DataFrame con operaciones
            
        Returns:
            Diccionario con máximos consecutivos
        """
        if trades.empty:
            return {'max_consecutive_wins': 0, 'max_consecutive_losses': 0}
        
        # Crear serie de resultados (1 para ganadoras, -1 para perdedoras)
        results = np.where(trades['profit_loss'] > 0, 1, -1)
        
        # Calcular rachas
        win_streaks = []
        loss_streaks = []
        
        current_streak = 0
        current_type = None
        
        for result in results:
            if current_type is None:
                # Primera operación
                current_streak = 1
                current_type = result
            elif result == current_type:
                # Continuar racha
                current_streak += 1
            else:
                # Fin de racha
                if current_type == 1:
                    win_streaks.append(current_streak)
                else:
                    loss_streaks.append(current_streak)
                
                # Iniciar nueva racha
                current_streak = 1
                current_type = result
        
        # Añadir última racha
        if current_type == 1:
            win_streaks.append(current_streak)
        else:
            loss_streaks.append(current_streak)
        
        # Obtener máximos
        max_consecutive_wins = max(win_streaks) if win_streaks else 0
        max_consecutive_losses = max(loss_streaks) if loss_streaks else 0
        
        return {
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses
        }
    
    @staticmethod
    def calculate_time_in_market(trades: pd.DataFrame, total_days: Optional[int] = None) -> float:
        """
        Calcula el porcentaje de tiempo en el mercado.
        
        Args:
            trades: DataFrame con operaciones
            total_days: Número total de días del período (opcional)
            
        Returns:
            Porcentaje de tiempo en el mercado
        """
        if trades.empty:
            return 0.0
        
        # Filtrar operaciones cerradas
        closed_trades = trades.dropna(subset=['exit_time'])
        
        if closed_trades.empty:
            return 0.0
        
        # Calcular duración total en mercado (en días)
        total_duration = closed_trades['duration'].apply(lambda x: x.total_seconds() / (24*3600)).sum()
        
        # Si no se proporciona el total de días, calcular desde la primera entrada a la última salida
        if total_days is None:
            first_entry = trades['entry_time'].min()
            last_exit = closed_trades['exit_time'].max()
            total_days = (last_exit - first_entry).total_seconds() / (24*3600)
        
        # Calcular porcentaje
        if total_days == 0:
            return 0.0
        
        time_in_market = total_duration / total_days * 100
        
        return min(time_in_market, 100.0)  # Limitar a 100%
    
    @staticmethod
    def calculate_avg_win_loss_ratio(trades: pd.DataFrame) -> float:
        """
        Calcula la relación entre la ganancia media y la pérdida media.
        
        Args:
            trades: DataFrame con operaciones
            
        Returns:
            Ratio ganancia/pérdida media
        """
        if trades.empty:
            return 0.0
        
        winning_trades = trades[trades['profit_loss'] > 0]
        losing_trades = trades[trades['profit_loss'] < 0]
        
        avg_win = winning_trades['profit_loss'].mean() if not winning_trades.empty else 0
        avg_loss = abs(losing_trades['profit_loss'].mean()) if not losing_trades.empty else 0
        
        if avg_loss == 0:
            return float('inf') if avg_win > 0 else 0.0
        
        return avg_win / avg_loss
    
    @staticmethod
    def calculate_all_metrics(trades: pd.DataFrame, 
                              equity_curve: pd.DataFrame, 
                              returns: pd.Series) -> Dict[str, Any]:
        """
        Calcula todas las métricas de rendimiento.
        
        Args:
            trades: DataFrame con operaciones
            equity_curve: DataFrame con la curva de equidad
            returns: Serie con retornos diarios
            
        Returns:
            Diccionario con todas las métricas
        """
        # Calcular drawdown
        dd_df, dd_metrics = MetricsCalculator.calculate_drawdown(equity_curve)
        
        # Calcular otras métricas
        win_rate = MetricsCalculator.calculate_win_rate(trades)
        profit_factor = MetricsCalculator.calculate_profit_factor(trades)
        expectancy = MetricsCalculator.calculate_expectancy(trades)
        sharpe_ratio = MetricsCalculator.calculate_sharpe_ratio(returns)
        sortino_ratio = MetricsCalculator.calculate_sortino_ratio(returns)
        cagr = MetricsCalculator.calculate_cagr(equity_curve)
        calmar_ratio = MetricsCalculator.calculate_calmar_ratio(returns, dd_metrics['max_drawdown_pct'])
        var_95 = MetricsCalculator.calculate_var(returns, 0.95)
        consecutive = MetricsCalculator.calculate_max_consecutive_wins_losses(trades)
        time_in_market = MetricsCalculator.calculate_time_in_market(trades)
        avg_win_loss_ratio = MetricsCalculator.calculate_avg_win_loss_ratio(trades)
        
        # Calcular métricas básicas de las operaciones
        total_trades = len(trades)
        winning_trades = len(trades[trades['profit_loss'] > 0])
        losing_trades = len(trades[trades['profit_loss'] < 0])
        avg_profit = trades['profit_loss'].mean() if not trades.empty else 0
        max_profit = trades['profit_loss'].max() if not trades.empty else 0
        max_loss = trades['profit_loss'].min() if not trades.empty else 0
        
        # Calcular estadísticas de duración
        if 'duration' in trades.columns and not trades.empty:
            avg_duration = pd.to_timedelta(trades['duration'].mean())
            max_duration = pd.to_timedelta(trades['duration'].max())
        else:
            avg_duration = pd.Timedelta(0)
            max_duration = pd.Timedelta(0)
        
        # Calcular estadísticas de retornos
        if not returns.empty:
            total_return = returns.sum() * 100
            annualized_return = returns.mean() * 252 * 100
            volatility = returns.std() * np.sqrt(252) * 100
        else:
            total_return = 0.0
            annualized_return = 0.0
            volatility = 0.0
        
        # Combinar todas las métricas
        all_metrics = {
            # Métricas generales
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate * 100,  # Convertir a porcentaje
            'profit_factor': profit_factor,
            'avg_profit': avg_profit,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'avg_win_loss_ratio': avg_win_loss_ratio,
            
            # Métricas de duración
            'avg_duration_days': avg_duration.total_seconds() / (24*3600),
            'max_duration_days': max_duration.total_seconds() / (24*3600),
            'time_in_market_pct': time_in_market,
            
            # Métricas de expectativa
            'expectancy': expectancy['expectancy'],
            'avg_win': expectancy['avg_win'],
            'avg_loss': expectancy['avg_loss'],
            'r_expectancy': expectancy['r_expectancy'],
            
            # Métricas de rachas
            'max_consecutive_wins': consecutive['max_consecutive_wins'],
            'max_consecutive_losses': consecutive['max_consecutive_losses'],
            
            # Métricas de riesgo/retorno
            'total_return_pct': total_return,
            'annualized_return_pct': annualized_return,
            'cagr_pct': cagr,
            'volatility_pct': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'var_95_pct': var_95,
            
            # Métricas de drawdown
            'max_drawdown_pct': dd_metrics['max_drawdown_pct'],
            'avg_drawdown_duration': dd_metrics['avg_drawdown_duration'],
            'max_drawdown_duration': dd_metrics['max_drawdown_duration'],
            'drawdown_periods': dd_metrics['drawdown_periods']
        }
        
        return all_metrics