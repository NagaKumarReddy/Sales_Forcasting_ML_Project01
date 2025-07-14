import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import itertools

st.set_page_config(page_title='Sales Forecasting App', layout='wide')
st.title('Sales Forecasting with Time Series Analysis')
st.write('Upload your sales data (Excel file) and get forecasts, diagnostics, and evaluation metrics!')

uploaded_file = st.file_uploader('Upload Excel file', type=['xlsx'])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader('Raw Data Preview')
    st.dataframe(df.head())

    # Preprocessing
    if 'Order Date' not in df.columns or 'Sales' not in df.columns:
        st.error("Dataset must contain 'Order Date' and 'Sales' columns.")
        st.stop()
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df.set_index('Order Date', inplace=True)
    df['Sales'] = df['Sales'].fillna(0)

    # EDA
    st.subheader('Exploratory Data Analysis')
    fig1, ax1 = plt.subplots()
    ax1.hist(df['Sales'], bins=50, color='skyblue', edgecolor='black')
    ax1.set_title('Sales Distribution')
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.boxplot(df['Sales'], vert=False)
    ax2.set_title('Boxplot of Sales')
    st.pyplot(fig2)

    if 'Category' in df.columns:
        fig3, ax3 = plt.subplots()
        df.groupby('Category')['Sales'].sum().plot(kind='bar', ax=ax3)
        ax3.set_title('Total Sales by Category')
        st.pyplot(fig3)

    # Monthly sales
    monthly_sales = df['Sales'].resample('M').sum()
    fig4, ax4 = plt.subplots(figsize=(10,4))
    ax4.plot(monthly_sales, label='Monthly Sales')
    ax4.set_title('Monthly Sales Over Time')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Sales')
    ax4.legend()
    st.pyplot(fig4)

    # Stationarity check
    st.subheader('Stationarity Check (ADF Test)')
    def adf_test(series):
        adf_result = adfuller(series.dropna(), autolag='AIC')
        adf_stat, p_value, usedlag, nobs = adf_result[:4]
        result = {'ADF Statistic': adf_stat, 'p-value': p_value, '# Lags Used': usedlag, 'N Obs': nobs}
        if len(adf_result) > 4:
            crit_values = adf_result[4]
            for key, val in crit_values.items():
                result[f'Critical Value ({key})'] = val
        return result
    st.write('ADF Test on Monthly Sales:', adf_test(monthly_sales))
    log_sales = np.log1p(monthly_sales)
    st.write('ADF Test on Log-Transformed Sales:', adf_test(log_sales))
    log_diff_sales = log_sales.diff().dropna()
    st.write('ADF Test on Log-Differenced Sales:', adf_test(log_diff_sales))
    fig5, ax5 = plt.subplots(figsize=(10,4))
    ax5.plot(log_diff_sales, label='Log-Differenced Sales')
    ax5.set_title('Log-Differenced Monthly Sales')
    ax5.legend()
    st.pyplot(fig5)

    # ARIMA parameter grid search
    st.subheader('ARIMA Model Selection')
    p = d = q = range(0, 3)
    pdq = list(itertools.product(p, d, q))
    best_aic = float('inf')
    best_order = None
    for order in pdq:
        try:
            model = ARIMA(log_diff_sales, order=order)
            model_fit = model.fit()
            if model_fit.aic < best_aic:
                best_aic = model_fit.aic
                best_order = order
        except:
            continue
    st.write(f'Best ARIMA order: {best_order} (AIC: {best_aic:.2f})')

    # Train/test split
    train = log_diff_sales[:-12]
    test = log_diff_sales[-12:]
    model = ARIMA(train, order=best_order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=12)
    forecast_cumsum = forecast.cumsum() + log_sales.iloc[-13]
    forecast_original = np.expm1(forecast_cumsum)
    test_cumsum = test.cumsum() + log_sales.iloc[-13]
    test_original = np.expm1(test_cumsum)

    # Forecast plot
    st.subheader('Forecast vs Actual')
    fig6, ax6 = plt.subplots(figsize=(10,4))
    ax6.plot(train.index, np.expm1(log_sales.loc[train.index]), label='Train')
    ax6.plot(test.index, test_original, label='Test')
    ax6.plot(test.index, forecast_original, label='Forecast', linestyle='--')
    ax6.set_title('Sales Forecast vs Actual')
    ax6.set_xlabel('Date')
    ax6.set_ylabel('Sales')
    ax6.legend()
    st.pyplot(fig6)

    # Evaluation metrics
    st.subheader('Evaluation Metrics')
    def mean_absolute_percentage_error(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mae = mean_absolute_error(test_original, forecast_original)
    rmse = np.sqrt(mean_squared_error(test_original, forecast_original))
    mape = mean_absolute_percentage_error(test_original, forecast_original)
    st.write(f'MAE: {mae:.2f}')
    st.write(f'RMSE: {rmse:.2f}')
    st.write(f'MAPE: {mape:.2f}%')

    # Residual diagnostics
    st.subheader('Model Diagnostics')
    residuals = model_fit.resid
    fig7, ax7 = plt.subplots(figsize=(10,4))
    ax7.plot(residuals)
    ax7.set_title('Residuals of ARIMA Model')
    st.pyplot(fig7)
    fig8, ax8 = plt.subplots(figsize=(10,4))
    plot_acf(residuals, lags=24, ax=ax8)
    ax8.set_title('ACF of Residuals')
    st.pyplot(fig8)
    fig9, ax9 = plt.subplots(figsize=(10,4))
    plot_pacf(residuals, lags=24, ax=ax9)
    ax9.set_title('PACF of Residuals')
    st.pyplot(fig9)
    st.info('If residuals are white noise (no autocorrelation), the model is well specified.')

    st.success('Analysis complete! Scroll up to review all results.')
else:
    st.info('Please upload an Excel file to begin.') 