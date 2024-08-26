import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go

import plotly.express as px
from prophet import Prophet

# Load the CSV file into a DataFrame
df = pd.read_csv("delhi_weather_2019_2024.csv")

# Convert 'Time' column to datetime format
df['Time'] = pd.to_datetime(df['Time'])
df.set_index('Time', inplace=True)

# Display the first few rows
print(df.head())

# Interactive plot for Temperature
fig_temp = px.line(df, x=df.index, y='Temperature (C)', 
                   title='Temperature in Delhi (January 2019 - August 2024)', 
                   labels={'x': 'Date', 'Temperature (C)': 'Temperature (C)'})
fig_temp.update_layout(yaxis=dict(range=[0, 55]), 
                       xaxis_title='Date', 
                       yaxis_title='Temperature (C)')
fig_temp.show()

# Interactive plot for Relative Humidity
fig_humidity = px.line(df, x=df.index, y='Relative Humidity (%)', 
                      title='Relative Humidity in Delhi (January 2019 - August 2024)', 
                      labels={'x': 'Date', 'Relative Humidity (%)': 'Relative Humidity (%)'})
fig_humidity.update_layout(yaxis=dict(range=[0, 120]), 
                           xaxis_title='Date', 
                           yaxis_title='Relative Humidity (%)')
fig_humidity.show()

# Interactive plot for Wind Speed
fig_windspeed = px.line(df, x=df.index, y='Wind Speed (m/s)', 
                       title='Wind Speed in Delhi (January 2019 - August 2024)', 
                       labels={'x': 'Date', 'Wind Speed (m/s)': 'Wind Speed (m/s)'})
fig_windspeed.update_layout(yaxis=dict(range=[0, df['Wind Speed (m/s)'].max() + 1]), 
                            xaxis_title='Date', 
                            yaxis_title='Wind Speed (m/s)')
fig_windspeed.show()

# Scatter plot for Temperature vs. Humidity
fig_temp_humidity = px.scatter(df, x='Temperature (C)', y='Relative Humidity (%)', 
                               title='Temperature vs. Humidity in Delhi (January 2019 - August 2024)', 
                               labels={'Temperature (C)': 'Temperature (C)', 'Relative Humidity (%)': 'Relative Humidity (%)'})
fig_temp_humidity.show()

# Scatter plot for Temperature vs. Wind Speed
fig_temp_windspeed = px.scatter(df, x='Temperature (C)', y='Wind Speed (m/s)', 
                                title='Temperature vs. Wind Speed in Delhi (January 2019 - August 2024)', 
                                labels={'Temperature (C)': 'Temperature (C)', 'Wind Speed (m/s)': 'Wind Speed (m/s)'})
fig_temp_windspeed.show()

# Scatter plot for Humidity vs. Wind Speed
fig_humidity_windspeed = px.scatter(df, x='Relative Humidity (%)', y='Wind Speed (m/s)', 
                                    title='Humidity vs. Wind Speed in Delhi (January 2019 - August 2024)', 
                                    labels={'Relative Humidity (%)': 'Relative Humidity (%)', 'Wind Speed (m/s)': 'Wind Speed (m/s)'})
fig_humidity_windspeed.show()

# Calculate correlation coefficients
temperature_humidity_corr = df['Temperature (C)'].corr(df['Relative Humidity (%)'])
temperature_windspeed_corr = df['Temperature (C)'].corr(df['Wind Speed (m/s)'])
humidity_windspeed_corr = df['Relative Humidity (%)'].corr(df['Wind Speed (m/s)'])

print(f"Correlation between Temperature and Humidity: {temperature_humidity_corr:.2f}")
print(f"Correlation between Temperature and Wind Speed: {temperature_windspeed_corr:.2f}")
print(f"Correlation between Humidity and Wind Speed: {humidity_windspeed_corr:.2f}")

# Forecasting function
# Prepare the data for Prophet
def prepare_data_for_prophet(df, column_name):
    df_prophet = df[[column_name]].reset_index()
    df_prophet.columns = ['ds', 'y']
    return df_prophet

# Forecasting function using Prophet and Plotly
def forecast_with_prophet(df, periods, title):
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    
    # Create interactive plot using Plotly
    fig = go.Figure()
    
    # Add the forecasted data
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'],
                             mode='lines', name='Forecast'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'],
                             mode='lines', name='Lower Bound', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'],
                             mode='lines', name='Upper Bound', line=dict(dash='dash')))
    
    fig.update_layout(title=title, 
                      xaxis_title='Date', 
                      yaxis_title='Value')
    
    # Show the plot
    fig.show()

# Number of days for 6 months forecast
num_days = 6 * 30  # Approximate number of days in 6 months

# Prepare data and forecast Temperature
df_temp = prepare_data_for_prophet(df, 'Temperature (C)')
forecast_with_prophet(df_temp, num_days, 'Temperature Forecast in Delhi (Next 6 Months)')

# Prepare data and forecast Relative Humidity
df_humidity = prepare_data_for_prophet(df, 'Relative Humidity (%)')
forecast_with_prophet(df_humidity, num_days, 'Relative Humidity Forecast in Delhi (Next 6 Months)')

# Prepare data and forecast Wind Speed
df_windspeed = prepare_data_for_prophet(df, 'Wind Speed (m/s)')
forecast_with_prophet(df_windspeed, num_days, 'Wind Speed Forecast in Delhi (Next 6 Months)')