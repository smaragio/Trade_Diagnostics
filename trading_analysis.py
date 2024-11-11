{\rtf1\ansi\ansicpg1252\cocoartf2820
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import pandas as pd\
import numpy as np\
from datetime import datetime\
import matplotlib.pyplot as plt\
from scipy import stats\
import os\
\
def preprocess_csv(df):\
    try:\
        df = df.copy()\
        \
        column_mappings = \{\
            'Date': 'Time', 'Timestamp': 'Time', 'DateTime': 'Time',\
            'Initial Balance': 'Balance Before', 'Starting Balance': 'Balance Before',\
            'Final Balance': 'Balance After', 'Ending Balance': 'Balance After',\
            'PnL': 'Realized P&L', 'Profit/Loss': 'Realized P&L',\
            'P&L': 'Realized P&L', 'PL': 'Realized P&L',\
            'Profit': 'Realized P&L', 'Realized PnL': 'Realized P&L',\
            'Realized P&L (value)': 'Realized P&L',\
            'Trade': 'Action', 'Description': 'Action', 'Trade Description': 'Action'\
        \}\
        \
        df.columns = df.columns.str.strip()\
        df.rename(columns=column_mappings, inplace=True)\
        \
        required_columns = ['Time', 'Balance Before', 'Balance After', 'Realized P&L', 'Action']\
        missing_columns = [col for col in required_columns if col not in df.columns]\
        \
        if missing_columns:\
            if 'Time' in missing_columns and 'Date' in df.columns:\
                df['Time'] = pd.to_datetime(df['Date'])\
            \
            if 'Balance Before' in missing_columns and 'Balance After' in df.columns and 'Realized P&L' in df.columns:\
                df['Balance Before'] = df['Balance After'] - df['Realized P&L']\
            \
            if 'Balance After' in missing_columns and 'Balance Before' in df.columns and 'Realized P&L' in df.columns:\
                df['Balance After'] = df['Balance Before'] + df['Realized P&L']\
            \
            if 'Realized P&L' in missing_columns and 'Balance Before' in df.columns and 'Balance After' in df.columns:\
                df['Realized P&L'] = df['Balance After'] - df['Balance Before']\
        \
        missing_columns = [col for col in required_columns if col not in df.columns]\
        if missing_columns:\
            raise ValueError(f"Unable to create required columns: \{missing_columns\}")\
        \
        df['Time'] = pd.to_datetime(df['Time'])\
        df['Balance Before'] = pd.to_numeric(df['Balance Before'], errors='coerce')\
        df['Balance After'] = pd.to_numeric(df['Balance After'], errors='coerce')\
        df['Realized P&L'] = pd.to_numeric(df['Realized P&L'], errors='coerce')\
        \
        if 'Action' in df.columns:\
            df['Action'] = df['Action'].astype(str)\
            \
            def standardize_action(action):\
                action = action.lower()\
                if 'close' not in action:\
                    if 'buy' in action or 'long' in action:\
                        return f"Close long position for \{action\}"\
                    elif 'sell' in action or 'short' in action:\
                        return f"Close short position for \{action\}"\
                return action\
            \
            df['Action'] = df['Action'].apply(standardize_action)\
        \
        return df, None\
        \
    except Exception as e:\
        return None, f"Error preprocessing CSV: \{str(e)\}"\
\
def import_trading_data(file_path):\
    try:\
        if not os.path.exists(file_path):\
            raise FileNotFoundError(f"Could not find file: \{file_path\}")\
            \
        if not file_path.lower().endswith('.csv'):\
            raise ValueError("File must be a CSV file")\
        \
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']\
        df = None\
        \
        for encoding in encodings:\
            try:\
                df = pd.read_csv(file_path, encoding=encoding)\
                break\
            except UnicodeDecodeError:\
                continue\
        \
        if df is None:\
            raise ValueError("Unable to read CSV file with any supported encoding")\
        \
        df, error_message = preprocess_csv(df)\
        \
        if error_message:\
            print(error_message)\
            return None\
            \
        if df is not None:\
            print(f"Successfully loaded and preprocessed \{len(df)\} trades from \{file_path\}")\
            return df\
        \
    except Exception as e:\
        print(f"An error occurred while importing the file: \{e\}")\
        return None\
\
def load_and_prepare_data(df):\
    """Prepare the data for analysis with improved position type extraction."""\
    df['Time'] = pd.to_datetime(df['Time'])\
    \
    # Improved position type extraction\
    def extract_position_type(action):\
        if pd.isna(action):\
            return None\
        action = str(action).lower()\
        if 'close long' in action:\
            return 'long'\
        elif 'close short' in action:\
            return 'short'\
        return None\
    \
    df['Symbol'] = df['Action'].str.extract(r'symbol (\\S+)')\
    df['Position Type'] = df['Action'].apply(extract_position_type)\
    df['Units'] = df['Action'].str.extract(r'for (\\d+\\.?\\d*)')\
    df['Close Price'] = df['Action'].str.extract(r'at price (\\d+\\.?\\d*)')\
    \
    df['Units'] = pd.to_numeric(df['Units'])\
    df['Close Price'] = pd.to_numeric(df['Close Price'])\
    \
    df = df.sort_values('Time')\
    df['Trade Number'] = range(1, len(df) + 1)\
    df['Cumulative P&L'] = df['Realized P&L'].cumsum()\
    df['Equity'] = df['Balance After']\
    \
    return df\
\
def calculate_basic_metrics(df):\
    total_trades = len(df)\
    winning_trades = len(df[df['Realized P&L'] > 0])\
    losing_trades = len(df[df['Realized P&L'] < 0])\
    \
    metrics = \{\
        'Total Trades': total_trades,\
        'Total P&L': df['Realized P&L'].sum(),\
        'Average P&L per Trade': df['Realized P&L'].mean(),\
        'Win Rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,\
        'Winning Trades': winning_trades,\
        'Losing Trades': losing_trades,\
        'Largest Win': df['Realized P&L'].max(),\
        'Largest Loss': df['Realized P&L'].min(),\
        'Profit Factor': abs(df[df['Realized P&L'] > 0]['Realized P&L'].sum() / \
                           df[df['Realized P&L'] < 0]['Realized P&L'].sum()) if df[df['Realized P&L'] < 0]['Realized P&L'].sum() != 0 else float('inf'),\
        'Average Winner': df[df['Realized P&L'] > 0]['Realized P&L'].mean() if len(df[df['Realized P&L'] > 0]) > 0 else 0,\
        'Average Loser': df[df['Realized P&L'] < 0]['Realized P&L'].mean() if len(df[df['Realized P&L'] < 0]) > 0 else 0\
    \}\
    return metrics\
\
def analyze_by_instrument(df):\
    instrument_analysis = []\
    \
    for symbol in df['Symbol'].unique():\
        symbol_data = df[df['Symbol'] == symbol]\
        total_trades = len(symbol_data)\
        winning_trades = len(symbol_data[symbol_data['Realized P&L'] > 0])\
        \
        metrics = \{\
            'Symbol': symbol,\
            'Total Trades': total_trades,\
            'Total P&L': symbol_data['Realized P&L'].sum(),\
            'Average P&L': symbol_data['Realized P&L'].mean(),\
            'Win Rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,\
            'Largest Win': symbol_data['Realized P&L'].max(),\
            'Largest Loss': symbol_data['Realized P&L'].min()\
        \}\
        instrument_analysis.append(metrics)\
    \
    return pd.DataFrame(instrument_analysis).sort_values('Total P&L', ascending=False)\
\
def analyze_position_types(df):\
    """Analyze performance by position type (long vs short)."""\
    position_analysis = []\
    \
    # Remove rows where Position Type is NaN\
    df_clean = df.dropna(subset=['Position Type'])\
    \
    # Only proceed if we have valid position types\
    if len(df_clean) > 0:\
        for pos_type in df_clean['Position Type'].unique():\
            pos_data = df_clean[df_clean['Position Type'] == pos_type]\
            total_trades = len(pos_data)\
            winning_trades = len(pos_data[pos_data['Realized P&L'] > 0])\
            \
            metrics = \{\
                'Position Type': pos_type,\
                'Total Trades': total_trades,\
                'Total P&L': pos_data['Realized P&L'].sum(),\
                'Average P&L': pos_data['Realized P&L'].mean(),\
                'Win Rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0\
            \}\
            position_analysis.append(metrics)\
    else:\
        # If no valid position types found, add all trades as "undefined"\
        total_trades = len(df)\
        winning_trades = len(df[df['Realized P&L'] > 0])\
        \
        metrics = \{\
            'Position Type': 'undefined',\
            'Total Trades': total_trades,\
            'Total P&L': df['Realized P&L'].sum(),\
            'Average P&L': df['Realized P&L'].mean(),\
            'Win Rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0\
        \}\
        position_analysis.append(metrics)\
    \
    return pd.DataFrame(position_analysis)\
\
\
def calculate_drawdown(df):\
    running_max = df['Equity'].expanding().max()\
    drawdown_dollars = df['Equity'] - running_max\
    drawdown_pct = (drawdown_dollars / running_max) * 100\
    max_drawdown_dollars = drawdown_dollars.min()\
    max_drawdown_pct = drawdown_pct.min()\
    \
    return drawdown_dollars, drawdown_pct, max_drawdown_dollars, max_drawdown_pct\
\
def run_diagnostics(df):\
    df = load_and_prepare_data(df)\
    \
    _, _, max_dd_dollars, max_dd_pct = calculate_drawdown(df)\
    \
    basic_metrics = calculate_basic_metrics(df)\
    basic_metrics['Maximum Drawdown ($)'] = max_dd_dollars\
    basic_metrics['Maximum Drawdown (%)'] = max_dd_pct\
    \
    results = \{\
        'basic_metrics': basic_metrics,\
        'instrument_analysis': analyze_by_instrument(df),\
        'position_analysis': analyze_position_types(df),\
        'df': df\
    \}\
    \
    return results\
\
file_path = input("Please paste your CSV file path here: ")\
file_path = file_path.strip('"').strip("'")\
df = import_trading_data(file_path)\
\
if df is not None:\
    results = run_diagnostics(df)\
    df = results['df']\
    \
    print("\\n=== Basic Performance Metrics ===")\
    for metric, value in results['basic_metrics'].items():\
        if isinstance(value, float):\
            print(f"\{metric\}: \{value:.2f\}")\
        else:\
            print(f"\{metric\}: \{value\}")\
    \
    print("\\n=== Instrument Analysis ===")\
    print(results['instrument_analysis'].to_string())\
    \
    print("\\n=== Position Type Analysis ===")\
    print(results['position_analysis'].to_string())\
    \
    plt.style.use('seaborn-v0_8-darkgrid')\
    fig = plt.figure(figsize=(20, 15))\
    \
    ax1 = plt.subplot(2, 2, 1)\
    ax1.plot(df['Trade Number'], df['Equity'], label='Account Equity', color='blue', linewidth=2)\
    z = np.polyfit(df['Trade Number'], df['Equity'], 1)\
    p = np.poly1d(z)\
    ax1.plot(df['Trade Number'], p(df['Trade Number']), '--', color='red', label='Trend', linewidth=1.5)\
    total_return = ((df['Equity'].iloc[-1] - df['Equity'].iloc[0]) / df['Equity'].iloc[0] * 100)\
    ax1.set_title(f'Equity Curve (Total Return: \{total_return:.1f\}%)', pad=20, fontsize=12)\
    ax1.set_xlabel('Trade #', fontsize=10)\
    ax1.set_ylabel('Equity ($)', fontsize=10)\
    ax1.grid(True, alpha=0.3)\
    ax1.legend(fontsize=10)\
    \
    ax2 = plt.subplot(2, 2, 2)\
    colors = np.where(df['Realized P&L'] >= 0, 'green', 'red')\
    ax2.scatter(df['Trade Number'], df['Realized P&L'], alpha=0.6, c=colors, s=50)\
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)\
    z = np.polyfit(df['Trade Number'], df['Realized P&L'], 1)\
    p = np.poly1d(z)\
    ax2.plot(df['Trade Number'], p(df['Trade Number']), '--', color='blue', alpha=0.5, label='Trend', linewidth=1.5)\
    ax2.set_title('Trade Returns', pad=20, fontsize=12)\
    ax2.set_xlabel('Trade #', fontsize=10)\
    ax2.set_ylabel('P&L ($)', fontsize=10)\
    ax2.grid(True, alpha=0.3)\
    ax2.legend(fontsize=10)\
    \
    ax3 = plt.subplot(2, 2, 3)\
    drawdown_dollars, drawdown_pct, max_dd_dollars, max_dd_pct = calculate_drawdown(df)\
    ax3.plot(df['Trade Number'], drawdown_pct, color='red', label='Drawdown %', linewidth=2)\
    ax3.fill_between(df['Trade Number'], drawdown_pct, 0, color='red', alpha=0.2)\
    ax3.set_title(f'Drawdown (Max: \{max_dd_pct:.1f\}%)', pad=20, fontsize=12)\
    ax3.set_xlabel('Trade #', fontsize=10)\
    ax3.set_ylabel('Drawdown (%)', fontsize=10)\
    ax3.grid(True, alpha=0.3)\
    ax3.legend(fontsize=10)\
    \
    ax4 = plt.subplot(2, 2, 4)\
    returns_data = df['Realized P&L'].replace([np.inf, -np.inf], np.nan).dropna()\
    \
    n, bins, patches = ax4.hist(returns_data, bins=30, density=True, alpha=0.6, color='blue', label='Returns')\
    \
    kde_x = np.linspace(returns_data.min(), returns_data.max(), 100)\
    kde = stats.gaussian_kde(returns_data)\
    ax4.plot(kde_x, kde(kde_x), color='navy', linewidth=1.5, label='KDE')\
    \
    ax4.axvline(returns_data.mean(), color='green', linestyle='--', \
                label=f'Mean: $\{returns_data.mean():.2f\}', linewidth=1.5)\
    ax4.axvline(returns_data.median(), color='red', linestyle='--', \
                label=f'Median: $\{returns_data.median():.2f\}', linewidth=1.5)\
    ax4.axvline(0, color='black', linestyle='-', alpha=0.3)\
    \
    skew = returns_data.skew()\
    kurt = returns_data.kurtosis()\
    ax4.set_title(f'Return Distribution\\nSkew: \{skew:.2f\}, Kurtosis: \{kurt:.2f\}', pad=20, fontsize=12)\
    ax4.set_xlabel('P&L ($)', fontsize=10)\
    ax4.set_ylabel('Density', fontsize=10)\
    ax4.grid(True, alpha=0.3)\
    ax4.legend(fontsize=10)\
    \
    plt.tight_layout(pad=3.0)\
    plt.show()\
else:\
    print("Unable to proceed with analysis due to data import error.")}