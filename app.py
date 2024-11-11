import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from trading_analysis import run_diagnostics, calculate_drawdown  # Import from your analysis file

def create_plots(df, results):
    """Create all plots in a single figure"""
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(20, 15))
    
    # Equity Curve (Top Left)
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(df['Trade Number'], df['Equity'], label='Account Equity', color='blue', linewidth=2)
    z = np.polyfit(df['Trade Number'], df['Equity'], 1)
    p = np.poly1d(z)
    ax1.plot(df['Trade Number'], p(df['Trade Number']), '--', color='red', label='Trend', linewidth=1.5)
    total_return = ((df['Equity'].iloc[-1] - df['Equity'].iloc[0]) / df['Equity'].iloc[0] * 100)
    ax1.set_title(f'Equity Curve (Total Return: {total_return:.1f}%)', pad=20, fontsize=12)
    ax1.set_xlabel('Trade #', fontsize=10)
    ax1.set_ylabel('Equity ($)', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Returns Scatter (Top Right)
    ax2 = plt.subplot(2, 2, 2)
    colors = np.where(df['Realized P&L'] >= 0, 'green', 'red')
    ax2.scatter(df['Trade Number'], df['Realized P&L'], alpha=0.6, c=colors, s=50)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    z = np.polyfit(df['Trade Number'], df['Realized P&L'], 1)
    p = np.poly1d(z)
    ax2.plot(df['Trade Number'], p(df['Trade Number']), '--', color='blue', alpha=0.5, label='Trend', linewidth=1.5)
    ax2.set_title('Trade Returns', pad=20, fontsize=12)
    ax2.set_xlabel('Trade #', fontsize=10)
    ax2.set_ylabel('P&L ($)', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Drawdown (Bottom Left)
    ax3 = plt.subplot(2, 2, 3)
    drawdown_dollars, drawdown_pct, max_dd_dollars, max_dd_pct = calculate_drawdown(df)
    ax3.plot(df['Trade Number'], drawdown_pct, color='red', label='Drawdown %', linewidth=2)
    ax3.fill_between(df['Trade Number'], drawdown_pct, 0, color='red', alpha=0.2)
    ax3.set_title(f'Drawdown (Max: {max_dd_pct:.1f}%)', pad=20, fontsize=12)
    ax3.set_xlabel('Trade #', fontsize=10)
    ax3.set_ylabel('Drawdown (%)', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    
    # Return Distribution (Bottom Right)
    ax4 = plt.subplot(2, 2, 4)
    returns_data = df['Realized P&L'].replace([np.inf, -np.inf], np.nan).dropna()
    
    n, bins, patches = ax4.hist(returns_data, bins=30, density=True, alpha=0.6, color='blue', label='Returns')
    
    kde_x = np.linspace(returns_data.min(), returns_data.max(), 100)
    kde = stats.gaussian_kde(returns_data)
    ax4.plot(kde_x, kde(kde_x), color='navy', linewidth=1.5, label='KDE')
    
    ax4.axvline(returns_data.mean(), color='green', linestyle='--', 
                label=f'Mean: ${returns_data.mean():.2f}', linewidth=1.5)
    ax4.axvline(returns_data.median(), color='red', linestyle='--', 
                label=f'Median: ${returns_data.median():.2f}', linewidth=1.5)
    ax4.axvline(0, color='black', linestyle='-', alpha=0.3)
    
    skew = returns_data.skew()
    kurt = returns_data.kurtosis()
    ax4.set_title(f'Return Distribution\nSkew: {skew:.2f}, Kurtosis: {kurt:.2f}', pad=20, fontsize=12)
    ax4.set_xlabel('P&L ($)', fontsize=10)
    ax4.set_ylabel('Density', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)
    
    plt.tight_layout(pad=3.0)
    return fig

def main():
    st.set_page_config(layout="wide")
    st.title('Trading Performance Analysis')
    
    uploaded_file = st.file_uploader("Upload your trading CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            results = run_diagnostics(df)
            
            # Display metrics
            st.header('Basic Performance Metrics')
            metrics_df = pd.DataFrame([results['basic_metrics']]).T
            metrics_df.columns = ['Value']
            st.dataframe(metrics_df)
            
            # Create two columns for analyses
            col1, col2 = st.columns(2)
            
            with col1:
                st.header('Instrument Analysis')
                st.dataframe(results['instrument_analysis'])
            
            with col2:
                st.header('Position Type Analysis')
                st.dataframe(results['position_analysis'])
            
            # Display plots
            st.header('Performance Visualization')
            fig = create_plots(results['df'], results)
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.write("Please make sure your CSV file has the correct format.")

if __name__ == '__main__':
    main()