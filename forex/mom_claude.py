import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FXMomentumBacktest:
    def __init__(self, lookback_months: int = 4, max_position_size: float = 0.25):
        """
        Initialize the FX Momentum strategy.
        
        Args:
            lookback_months: Number of months to look back for trend analysis
            max_position_size: Maximum position size as a fraction of portfolio
        """
        self.lookback_months = lookback_months
        self.max_position_size = max_position_size
        self.currencies = ['USD', 'EUR', 'JPY', 'CNY']
        self.pairs = self._generate_currency_pairs()
        self.rates = None
        
    def _generate_currency_pairs(self) -> List[str]:
        """Generate all possible currency pairs from the currency list."""
        pairs = []
        for i, base in enumerate(self.currencies):
            for quote in self.currencies[i+1:]:
                pairs.append(f"{base}/{quote}")
        return pairs
    
    def fetch_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch FX rates with proper error handling and data validation.
        
        Args:
            start_date: Start date for data fetching
            end_date: End date for data fetching
            
        Returns:
            DataFrame containing FX rates
        """
        try:
            # Add extra months before start_date for lookback period
            fetch_start = (pd.to_datetime(start_date) - relativedelta(months=self.lookback_months + 1)).strftime('%Y-%m-%d')
            
            fx_data: Dict[str, Optional[pd.Series]] = {}
            yf_symbols = {
                'USD/EUR': 'USDEUR=X',
                'USD/JPY': 'USDJPY=X',
                'USD/CNY': 'USDCNY=X',
                'EUR/JPY': 'EURJPY=X',
                'EUR/CNY': 'EURCNY=X',
                'JPY/CNY': None
            }
            
            for pair, symbol in yf_symbols.items():
                if symbol is not None:
                    try:
                        ticker = yf.Ticker(symbol)
                        data = ticker.history(start=fetch_start, end=end_date, interval='1mo')
                        
                        # Validate data
                        if data.empty:
                            logger.warning(f"No data received for {pair}")
                            fx_data[pair] = None
                            continue
                            
                        # Use previous day's close for month-end to avoid look-ahead bias
                        fx_data[pair] = data['Close'].resample('M').asfreq()
                        
                        # Forward fill missing values (max 2 periods)
                        fx_data[pair] = fx_data[pair].fillna(method='ffill', limit=2)
                        
                    except Exception as e:
                        logger.error(f"Error fetching data for {pair}: {e}")
                        fx_data[pair] = None
            
            # Calculate cross-rates without look-ahead bias
            if fx_data['USD/JPY'] is not None and fx_data['USD/CNY'] is not None:
                fx_data['JPY/CNY'] = fx_data['USD/CNY'] / fx_data['USD/JPY']
            
            # Create DataFrame and validate
            self.rates = pd.DataFrame(fx_data)
            if self.rates.empty:
                raise ValueError("No valid data fetched for any currency pair")
            
            # Check for minimum required data
            min_required = self.lookback_months + 2  # +2 for signal generation and trading
            if len(self.rates) < min_required:
                raise ValueError(f"Insufficient data points. Need at least {min_required} months.")
            
            return self.rates
            
        except Exception as e:
            logger.error(f"Fatal error in data fetching: {e}")
            raise
    
    def _calculate_position_size(self, capital: float) -> float:
        """Calculate position size with proper risk management."""
        return capital * self.max_position_size
    
    def _check_trend(self, rates: pd.DataFrame, pair: str, current_date: pd.Timestamp) -> Tuple[int, float]:
        """
        Check trend using only data available at current_date.
        Returns both signal and strength of trend.
        """
        try:
            # Get data up to previous month to avoid look-ahead bias
            previous_month = current_date - pd.DateOffset(months=1)
            available_data = rates.loc[:previous_month]
            
            if len(available_data) < self.lookback_months + 1:
                return 0, 0.0
            
            historical_data = available_data.iloc[-self.lookback_months-1:-1]
            current_rate = available_data.iloc[-1]
            
            # Calculate trend strength using z-score
            returns = np.log(current_rate[pair] / historical_data[pair])
            trend_strength = (returns.mean() / returns.std()) if returns.std() != 0 else 0
            
            # Determine signal with minimum strength threshold
            if trend_strength > 1.0:
                return 1, abs(trend_strength)
            elif trend_strength < -1.0:
                return -1, abs(trend_strength)
            
            return 0, 0.0
            
        except Exception as e:
            logger.error(f"Error in trend calculation for {pair}: {e}")
            return 0, 0.0
    
    def run_backtest(self, start_date: str, initial_capital: float = 1000000) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run backtest with proper risk management and transaction costs.
        """
        try:
            # Initialize results storage
            portfolio_values = []
            monthly_returns = []
            positions = pd.DataFrame(0, index=self.rates.index, columns=self.pairs)
            exposure = pd.DataFrame(0, index=self.rates.index, columns=self.currencies)
            
            # Convert start_date to timestamp
            start_ts = pd.to_datetime(start_date)
            
            # Track portfolio value
            current_capital = initial_capital
            
            # Trading costs
            TRANSACTION_COST = 0.0001  # 1 bp per trade
            
            # Iterate through months
            trading_dates = self.rates.index[self.rates.index >= start_ts]
            
            for i in range(len(trading_dates)):
                current_date = trading_dates[i]
                
                if i < self.lookback_months:
                    portfolio_values.append(current_capital)
                    monthly_returns.append(0.0)
                    continue
                
                # Determine positions
                current_exposure = {curr: 0 for curr in self.currencies}
                total_active_positions = 0
                
                for pair in self.pairs:
                    trend, strength = self._check_trend(self.rates.iloc[:i+1], pair, current_date)
                    
                    # Apply position sizing based on trend strength
                    position_size = self._calculate_position_size(current_capital) * min(1.0, strength)
                    positions.loc[current_date, pair] = trend * position_size
                    
                    if trend != 0:
                        base, quote = pair.split('/')
                        current_exposure[base] += trend * position_size
                        current_exposure[quote] -= trend * position_size
                        total_active_positions += 1
                
                # Record exposure
                for curr in self.currencies:
                    exposure.loc[current_date, curr] = current_exposure[curr]
                
                # Calculate returns with transaction costs
                if i > self.lookback_months:
                    period_returns = []
                    for pair in self.pairs:
                        prev_position = positions.iloc[-2][pair]
                        curr_position = positions.iloc[-1][pair]
                        
                        if prev_position != 0:
                            prev_rate = self.rates.iloc[i-1][pair]
                            current_rate = self.rates.iloc[i][pair]
                            
                            # Apply transaction costs if position changed
                            transaction_cost = TRANSACTION_COST if prev_position != curr_position else 0
                            pair_return = (current_rate / prev_rate - 1) * prev_position/current_capital - transaction_cost
                            period_returns.append(pair_return)
                    
                    if period_returns:
                        month_return = np.sum(period_returns)
                        current_capital *= (1 + month_return)
                        monthly_returns.append(month_return)
                    else:
                        monthly_returns.append(0.0)
                else:
                    monthly_returns.append(0.0)
                
                portfolio_values.append(current_capital)
            
            # Create results DataFrame
            results = pd.DataFrame({
                'Portfolio_Value': portfolio_values,
                'Returns': monthly_returns
            }, index=trading_dates)
            
            results['Cumulative_Returns'] = (1 + results['Returns']).cumprod() - 1
            results['Peak'] = results['Portfolio_Value'].expanding().max()
            results['Drawdown'] = (results['Portfolio_Value'] - results['Peak']) / results['Peak']
            
            return results, positions, exposure
            
        except Exception as e:
            logger.error(f"Error in backtest execution: {e}")
            raise

    def calculate_metrics(self, results: pd.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics with proper risk-adjusted measures."""
        try:
            metrics = {}
            
            # Basic metrics
            total_years = (results.index[-1] - results.index[0]).days / 365
            total_return = results['Portfolio_Value'].iloc[-1] / results['Portfolio_Value'].iloc[0] - 1
            metrics['Annual_Return'] = (1 + total_return) ** (1 / total_years) - 1
            
            # Risk metrics
            returns = results['Returns']
            metrics['Annual_Volatility'] = returns.std() * np.sqrt(12)
            metrics['Sharpe_Ratio'] = metrics['Annual_Return'] / metrics['Annual_Volatility'] if metrics['Annual_Volatility'] != 0 else 0
            metrics['Max_Drawdown'] = results['Drawdown'].min()
            
            # Additional risk metrics
            metrics['Sortino_Ratio'] = metrics['Annual_Return'] / (returns[returns < 0].std() * np.sqrt(12)) if len(returns[returns < 0]) > 0 else 0
            metrics['Win_Rate'] = len(returns[returns > 0]) / len(returns[returns != 0]) if len(returns[returns != 0]) > 0 else 0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            raise

    def plot_results(self, results: pd.DataFrame, positions: pd.DataFrame, exposure: pd.DataFrame) -> None:
        """Plot strategy results with enhanced visualizations."""
        try:
            plt.style.use('seaborn')
            
            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot cumulative returns
            results['Cumulative_Returns'].plot(ax=ax1)
            ax1.set_title('Cumulative Returns')
            ax1.set_ylabel('Return')
            ax1.grid(True)
            
            # Plot drawdowns
            results['Drawdown'].plot(ax=ax2, color='red')
            ax2.set_title('Drawdowns')
            ax2.set_ylabel('Drawdown')
            ax2.grid(True)
            
            # Plot rolling volatility
            rolling_vol = results['Returns'].rolling(window=12).std() * np.sqrt(12)
            rolling_vol.plot(ax=ax3)
            ax3.set_title('Rolling 12-Month Volatility')
            ax3.grid(True)
            
            # Plot rolling Sharpe ratio
            rolling_returns = results['Returns'].rolling(window=12).mean() * 12
            rolling_sharpe = rolling_returns / rolling_vol
            rolling_sharpe.plot(ax=ax4)
            ax4.set_title('Rolling 12-Month Sharpe Ratio')
            ax4.grid(True)
            
            plt.tight_layout()
            plt.show()
            
            # Plot currency exposure heatmap
            plt.figure(figsize=(12, 6))
            sns.heatmap(exposure.T, cmap='RdYlBu', center=0)
            plt.title('Currency Exposure Over Time')
            plt.show()
            
        except Exception as e:
            logger.error(f"Error in plotting results: {e}")
            raise

def main():
    try:
        # Set up the backtest
        start_date = '2020-01-01'
        end_date = '2023-12-31'
        initial_capital = 1000000
        
        # Create strategy
        strategy = FXMomentumBacktest(lookback_months=4, max_position_size=0.25)
        
        # Fetch data
        logger.info("Fetching FX data...")
        rates = strategy.fetch_data(start_date, end_date)
        
        # Run backtest
        logger.info("Running backtest...")
        results, positions, exposure = strategy.run_backtest(start_date, initial_capital)
        
        # Calculate and print metrics
        metrics = strategy.calculate_metrics(results)
        
        logger.info("\nStrategy Performance Metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.2%}")
        
        # Plot results
        strategy.plot_results(results, positions, exposure)
        
        # Print final portfolio value
        final_value = results['Portfolio_Value'].iloc[-1]
        logger.info(f"\nFinal Portfolio Value: ${final_value:,.2f}")
        
        # Print final exposures
        logger.info("\nFinal Currency Exposures:")
        logger.info(exposure.iloc[-1])
        
    except Exception as e:
        logger.error(f"Fatal error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()