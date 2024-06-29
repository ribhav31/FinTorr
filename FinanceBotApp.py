import streamlit as st
import datetime
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import yfinance as yf
import ta
import mplfinance as mpf
import plotly.graph_objects as go
import google.generativeai as palm

# Configure Palm API with your key
palm.configure(api_key='your api key here')

# models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
# selected_model = st.sidebar.selectbox("Select Chatbot Model", models, format_func=lambda x: x.name)

# Polygon.io API credentials
API_KEY = "your api key here"


def fetch_news(stock_symbol, num_articles=5):
    news_api_key = "YOUR_NEWS_API_KEY"  # Replace with your News API key
    today = datetime.date.today()
    one_week_ago = today - datetime.timedelta(days=7)
    endpoint = f"https://newsapi.org/v2/everything?q={stock_symbol}&from={one_week_ago}&to={today}&sortBy=popularity&apiKey={news_api_key}&language=en"
    response = requests.get(endpoint)
    if response.status_code == 200:
        data = response.json()
        articles = data.get('articles', [])
        return articles[:num_articles]
    else:
        return None
    
def generate_notifications(rsi, threshold_low, threshold_high):
    notifications = []
    for i in range(len(rsi)):
        if rsi[i] < threshold_low:
            notifications.append(f"RSI too low at index {i}")
        elif rsi[i] > threshold_high:
            notifications.append(f"RSI too high at index {i}")
    return notifications

# Function to perform sentiment analysis on news articles
def analyze_news_sentiment(articles):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = []
    for article in articles:
        title = article.get('title', '')
        description = article.get('description', '')
        text = title + ' ' + description
        sentiment = sia.polarity_scores(text)['compound']
        sentiment_scores.append(sentiment)
    return sentiment_scores

# Function to fetch historical stock data from Polygon.io API
def fetch_historical_data(symbol, start_date, end_date):
    endpoint = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}?apiKey={API_KEY}"
    response = requests.get(endpoint)
    if response.status_code == 200:
        data = response.json()
        if 'results' in data:
            return data['results']
    return None

def fetch_historical_data1(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

# Function to simulate a simple trading strategy
def simulate_trading_strategy(data):
    # Calculate moving averages (50-day and 200-day)
    data['MA50'] = data['c'].rolling(window=50).mean()
    data['MA200'] = data['c'].rolling(window=200).mean()

    # Generate buy/sell signals
    data['Signal'] = 0
    data['Signal'][50:] = (data['MA50'][50:] > data['MA200'][50:]).astype(int)

    return data

def fetch_realtime_data(symbol):
    # Get real-time data
    realtime_data = yf.download(symbol, start=datetime.datetime.now() - datetime.timedelta(days=1), end=datetime.datetime.now(), interval='1m')
    return realtime_data

def plot_realtime_data(realtime_data, symbol):
    if not realtime_data.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(realtime_data['Close'], label=symbol)
        plt.title(f'Real-time Data for {symbol}')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot()
    else:
        st.write('No real-time data available for the specified symbol.')

def fetch_historical_data2(symbol, period='1y'):
    # Fetch historical data
    historical_data = yf.download(symbol, period=period)
    return historical_data

def display_index_info(index_symbol):
    st.title(f"Finance App: {index_symbol}")

    # Fetch historical index data
    index_data = fetch_historical_data2(index_symbol)

    # Display historical closing prices
    st.subheader('Historical Prices')
    st.line_chart(index_data['Close'])

    # Display model accuracy - You may need to adapt this part depending on how you obtain accuracy for indices
    st.subheader('Model Accuracy:')
    st.write('Accuracy information for indices may not be directly applicable.')

    # Display summary information (You may need to adapt this based on available data)
    st.subheader('Summary')
    st.write('Summary information for indices may not be available.')

    # Display key statistics (You may need to adapt this based on available data)
    st.subheader('Key Statistics')
    st.write('Key statistics for indices may not be available.')

    # Display dividends (You may need to adapt this based on available data)
    st.subheader('Dividends')
    st.write('Dividend information for indices may not be available.')

    # Display additional fundamental information (You may need to adapt this based on available data)
    st.subheader('Additional Fundamental Information')
    st.write('Additional fundamental information for indices may not be available.')

    st.title('Index News Analysis')

    # Fetch news articles for the index
    articles = fetch_news(index_symbol)

    if articles:
        # Display news articles
        st.subheader('Latest News Articles')
        for article in articles:
            st.write(f"**{article['title']}**")
            st.write(article['description'])
            st.write(f"Source: {article['source']['name']}")
            st.write(f"Published At: {article['publishedAt']}")
            st.write("---")

        # Perform sentiment analysis (You may need to adapt this based on available data)
        sentiment_scores = analyze_news_sentiment(articles)

        # Calculate average sentiment score
        avg_sentiment = np.mean(sentiment_scores)

        # Display sentiment analysis
        st.subheader('News Sentiment Analysis')
        st.write(f"Average Sentiment Score: {avg_sentiment:.2f}")

        # Visualize sentiment scores
        sentiment_df = pd.DataFrame({'Sentiment Score': sentiment_scores})
        st.line_chart(sentiment_df)

    else:
        st.write('No news articles found for the specified index symbol.')


def plot_historical_line_chart(realtime_data):
    if not realtime_data.empty:
        st.subheader('Historical Prices')
        st.line_chart(realtime_data['Close'])
    else:
        st.write('No historical data available for the specified symbol.')

# Function to calculate portfolio value
def calculate_portfolio_value(df, initial_investment, buy_quantity, sell_quantity):
    portfolio_value = initial_investment
    if df is not None:
        portfolio_value += (buy_quantity - sell_quantity) * df['c'].iloc[-1]
    return portfolio_value

# Function to generate text using Palm API
def generate_text(prompt, model, temperature=0):
    completion = palm.generate_text(model=model, prompt=prompt, temperature=temperature, max_output_tokens=800)
    return completion.result

# Function to preprocess stock data
def preprocess_stock_data(stock_data):
    stock_data['Date'] = pd.to_datetime(stock_data.index)
    stock_data['DayOfWeek'] = stock_data['Date'].dt.dayofweek
    stock_data['Hour'] = stock_data['Date'].dt.hour
    stock_data['Minute'] = stock_data['Date'].dt.minute
    sia = SentimentIntensityAnalyzer()
    stock_data['Sentiment'] = stock_data['Date'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
    stock_data['Label'] = np.where(stock_data['Close'].shift(-1) > stock_data['Close'], 1, 0)
    return stock_data

# Function to extract features for machine learning
def extract_features(data):
    features = data[['Open', 'High', 'Low', 'Close', 'Volume', 'DayOfWeek', 'Hour', 'Minute', 'Sentiment']]
    return features

# Function to train a RandomForestClassifier
def train_model(features, labels):
    if len(features) < 2:
        raise ValueError("Insufficient data for training. Please provide more data.")

    test_size = min(0.2, len(features) - 1)  # Adjust test size based on the dataset size
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42)
    
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Unable to perform train-test split. Please check your data.")

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return clf, accuracy

# Function to plot historical closing prices
def plot_stock_prices(data, stock_name):
    st.subheader(f'{stock_name} Historical Closing Prices')
    plt.figure(figsize=(10, 5))
    plt.plot(data['Close'], label='Closing Price')
    plt.title(f'{stock_name} Historical Closing Prices')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    st.pyplot(plt)

# def calculate_rsi(data, window=14):
#     delta = data['c'].diff(1)
#     gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
#     loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
#     rs = gain / loss
#     rsi = 100 - (100 / (1 + rs))
#     return rsi

def calculate_rsi(df):
    """Calculate Relative Strength Index (RSI)"""
    indicator_rsi = ta.momentum.RSIIndicator(df['c'])
    df['RSI'] = indicator_rsi.rsi()
    return df

def calculate_rsi1(df, window=14):
    """Calculate Relative Strength Index (RSI)"""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(df):
    """Calculate Bollinger Bands"""
    indicator_bb = ta.volatility.BollingerBands(df['c'])
    df['Bollinger_Bands'] = indicator_bb.bollinger_hband(), indicator_bb.bollinger_mavg(), indicator_bb.bollinger_lband()
    return df

def calculate_macd(df):
    """Calculate Moving Average Convergence Divergence (MACD)"""
    indicator_macd = ta.trend.MACD(df['c'])
    df['MACD'] = indicator_macd.macd()
    df['Signal_Line'] = indicator_macd.macd_signal()
    return df

def plot_moving_averages(df):
    # Plot closing prices and moving averages
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df['c'], label='Close Price', color='blue')
    ax.plot(df.index, df['MA50'], label='50-Day MA', color='orange')
    ax.plot(df.index, df['MA200'], label='200-Day MA', color='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend(loc='upper left')
    plt.title('Stock Price and Moving Averages')
    return fig

def plot_rsi(df):
    # Plot RSI
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df['RSI'], label='RSI', color='purple')
    ax.axhline(70, color='red', linestyle='--')
    ax.axhline(30, color='green', linestyle='--')
    ax.set_ylabel('RSI')
    ax.set_xlabel('Date')
    ax.legend()
    plt.title('Relative Strength Index (RSI)')
    return fig

def calculate_rsi1(df, window=14):
    """Calculate Relative Strength Index (RSI)"""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_notifications(rsi, threshold_low, threshold_high):
    notifications = []
    for i in range(len(rsi)):
        if rsi[i] < threshold_low:
            notifications.append(f"RSI too low at index {i}")
        elif rsi[i] > threshold_high:
            notifications.append(f"RSI too high at index {i}")
    return notifications

# def plot_bollinger_bands(df):
#     # Plot Bollinger Bands
#     fig, ax = plt.subplots(figsize=(12, 4))
#     ax.plot(df.index, df['Bollinger_Bands'][0], label='Upper Bollinger Band', color='green')
#     ax.plot(df.index, df['Bollinger_Bands'][1], label='Middle Bollinger Band', color='blue')
#     ax.plot(df.index, df['Bollinger_Bands'][2], label='Lower Bollinger Band', color='red')
#     ax.set_xlabel('Date')
#     ax.set_ylabel('Price')
#     ax.legend()
#     plt.title('Bollinger Bands')
#     return fig

def plot_macd(df):
    # Plot MACD
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df['MACD'], label='MACD', color='orange')
    ax.plot(df.index, df['Signal_Line'], label='Signal Line', color='purple', linestyle='--')
    ax.set_xlabel('Date')
    ax.set_ylabel('MACD')
    ax.legend()
    plt.title('Moving Average Convergence Divergence (MACD)')
    return fig

# def convert_currency(amount, from_currency, to_currency):
#     # Fetch exchange rate data
#     exchange_rate = yf.download(f'{from_currency}{to_currency}=X')['Close'].iloc[-1]
    
#     # Convert currency
#     converted_amount = amount * exchange_rate
#     return converted_amount

def fetch_exchange_rate(from_currency, to_currency):
    # Fetch exchange rate data
    exchange_data = yf.download(f'{from_currency}{to_currency}=X')
    return exchange_data

def convert_currency(amount, exchange_data):
    # Perform currency conversion
    converted_amount = amount * exchange_data['Close']
    return converted_amount

def plot_exchange_rate(exchange_data, from_currency, to_currency):
    # Plot exchange rate
    plt.figure(figsize=(10, 6))
    plt.plot(exchange_data.index, exchange_data['Close'])
    plt.title(f'Exchange Rate: {from_currency}/{to_currency}')
    plt.xlabel('Date')
    plt.ylabel('Exchange Rate')
    plt.grid(True)
    st.pyplot()

# Function to display fundamental information
def display_fundamental_info(stock_info):
    st.subheader("Fundamental Information")

    # Display stock details
    st.write("Stock Details")
    st.write(stock_info)
    # Extract relevant fundamental data
    market_cap = stock_info.get('market_cap', 'N/A')
    dividend_yield = stock_info.get('dividend_yield', 'N/A')
    pe_ratio = stock_info.get('pe_ratio', 'N/A')

    # Display stock details
    st.write("Stock Market Cap:", market_cap)
    st.write("Stock Dividend Yield:", dividend_yield)
    st.write("Stock P/E Ratio:", pe_ratio)

    # Create and display graphs
    st.write("Fundamental Data Visualization")

    # Example: Market Cap Pie Chart
    labels = ['Large Cap', 'Mid Cap', 'Small Cap']
    sizes = [60, 30, 10]  # Example data, you can replace with actual data
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig1)

    # Example: Dividend Yield Bar Chart
    companies = ['Company A', 'Company B', 'Company C']
    div_yield = [3.5, 2.8, 4.2]  # Example data, you can replace with actual data
    plt.bar(companies, div_yield)
    plt.xlabel('Company')
    plt.ylabel('Dividend Yield (%)')
    plt.title('Dividend Yield Comparison')
    st.pyplot(plt)

import mplfinance as mpf

def display_stock_info(stock_symbol, selected_model):
    st.title(f"Finance App: {stock_symbol}")

    # Download historical stock data
    stock_data = yf.download(stock_symbol, start='2020-01-01', end='2023-01-01')

    # Preprocess stock data
    processed_data = preprocess_stock_data(stock_data)

    # Features for machine learning
    features = extract_features(processed_data)

    # Train a RandomForestClassifier
    labels = processed_data['Label']
    model, accuracy = train_model(features, labels)

    # Display historical closing prices
    plot_stock_prices(processed_data, stock_symbol)

    # Display candlestick chart
    st.subheader('Candlestick Chart')
    fig = go.Figure(data=[go.Candlestick(x=processed_data.index,
                                         open=processed_data['Open'],
                                         high=processed_data['High'],
                                         low=processed_data['Low'],
                                         close=processed_data['Close'])])
    st.plotly_chart(fig)

    # Display model accuracy
    st.subheader(f'Model Accuracy: {accuracy:.2f}')

    # Display fundamental information
    # stock_info = yf.Ticker(stock_symbol)
    # display_fundamental_info(stock_info.info)

    # Display in presentable manner
    stock_info = yf.Ticker(stock_symbol)

    # Display company name and logo
    if 'logo_url' in stock_info.info:
        st.image(stock_info.info['logo_url'])
    else:
        st.write("Logo URL not available for this stock.")
    st.header(stock_info.info['longName'])

    # Display summary information
    st.subheader('Summary')
    st.write(stock_info.info['longBusinessSummary'])

    # Display key statistics
    st.subheader('Key Statistics')
    if 'defaultKeyStatistics' in stock_info.info:
        # Display default key statistics in a table
        st.table(stock_info.info['defaultKeyStatistics'])
    else:
        st.write("Default key statistics not available for this stock.")

    # Display historical prices
    st.subheader('Historical Prices')
    historical_data = stock_info.history(period='1y')
    st.line_chart(historical_data['Close'])

    # Display dividends
    st.subheader('Dividends')
    dividends = stock_info.dividends
    if not dividends.empty:
        st.table(dividends)
    else:
        st.write('No dividend data available.')
    
    # display_fundamental_info(stock_info.info)
    company_profile = stock_info.info

    # Extract relevant information
    full_time_employees = company_profile.get("fullTimeEmployees")
    company_officers = company_profile.get("companyOfficers")

    for officer in company_officers:
        name = officer.get("name")
        title = officer.get("title")
        st.write("Name:", name)
        st.write("Title:", title)
        st.write()  # Add an empty line for readability

    audit_risk = stock_info.info.get("auditRisk")
    board_risk = stock_info.info.get("boardRisk")
    compensation_risk = stock_info.info.get("compensationRisk")
    shareholder_rights_risk = stock_info.info.get("shareHolderRightsRisk")
    overall_risk = stock_info.info.get("overallRisk")
    governance_epoch_date = stock_info.info.get("governanceEpochDate")
    compensation_as_of_epoch_date = stock_info.info.get("compensationAsOfEpochDate")
    ir_website = stock_info.info.get("irWebsite")

    st.subheader('Additional Fundamental Information')
    st.write(f"Audit Risk: {company_profile.get('auditRisk')}")
    st.write(f"Board Risk: {company_profile.get('boardRisk')}")
    st.write(f"Compensation Risk: {company_profile.get('compensationRisk')}")
    st.write(f"Shareholder Rights Risk: {company_profile.get('shareHolderRightsRisk')}")
    st.write(f"Overall Risk: {company_profile.get('overallRisk')}")
    st.write(f"Governance Epoch Date: {company_profile.get('governanceEpochDate')}")
    st.write(f"Compensation As Of Epoch Date: {company_profile.get('compensationAsOfEpochDate')}")
    st.write(f"IR Website: {company_profile.get('irWebsite')}")
    
    st.title('Stock News Analysis')

    # Sidebar input for stock symbol
    # Sidebar input for stock symbol
    stock_symbol = st.sidebar.text_input('Enter Stock Symbol', 'AAPL', key='stock_symbol_input')


    # Fetch news articles
    articles = fetch_news(stock_symbol)
    
    if articles:
        # Display news articles
        st.subheader('Latest News Articles')
        for article in articles:
            st.write(f"**{article['title']}**")
            st.write(article['description'])
            st.write(f"Source: {article['source']['name']}")
            st.write(f"Published At: {article['publishedAt']}")
            st.write("---")

        # Perform sentiment analysis
        sentiment_scores = analyze_news_sentiment(articles)

        # Calculate average sentiment score
        avg_sentiment = np.mean(sentiment_scores)

        # Display sentiment analysis
        st.subheader('News Sentiment Analysis')
        st.write(f"Average Sentiment Score: {avg_sentiment:.2f}")

        # Visualize sentiment scores
        sentiment_df = pd.DataFrame({'Sentiment Score': sentiment_scores})
        st.line_chart(sentiment_df)

    else:
        st.write('No news articles found for the specified stock symbol.')

    # Make predictions on new data
    new_data = features.iloc[-1, :].values.reshape(1, -1)
    prediction = model.predict(new_data)
    st.subheader(f'Predicted Label for {stock_symbol}: {prediction[0]}')

    # Generate investment recommendation
    if prediction[0] == 1:
        st.success("The model predicts that the stock price will go up. Consider investing.")
    else:
        st.warning("The model predicts that the stock price will go down. Consider caution.")
    
def plot_candlestick_chart(historical_data):
    fig = go.Figure(data=[go.Candlestick(x=historical_data.index,
                                          open=historical_data['Open'],
                                          high=historical_data['High'],
                                          low=historical_data['Low'],
                                          close=historical_data['Close'])])

    fig.update_layout(title='Candlestick Chart - Historical Prices',
                      xaxis_title='Date',
                      yaxis_title='Price')
    
    st.plotly_chart(fig)

def fetch_indian_indices_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Function to preprocess stock data
def preprocess_stock_data(stock_data):
    stock_data['Date'] = pd.to_datetime(stock_data.index)
    stock_data['DayOfWeek'] = stock_data['Date'].dt.dayofweek
    stock_data['Hour'] = stock_data['Date'].dt.hour
    stock_data['Minute'] = stock_data['Date'].dt.minute
    sia = SentimentIntensityAnalyzer()
    stock_data['Sentiment'] = stock_data['Date'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
    stock_data['Label'] = np.where(stock_data['Close'].shift(-1) > stock_data['Close'], 1, 0)
    return stock_data

# Function to extract features for machine learning
def extract_features(data):
    features = data[['Open', 'High', 'Low', 'Close', 'Volume', 'DayOfWeek', 'Hour', 'Minute', 'Sentiment']]
    return features

# Function to train a RandomForestClassifier
def train_model(features, labels):
    if len(features) < 2:
        raise ValueError("Insufficient data for training. Please provide more data.")

    test_size = min(0.2, len(features) - 1)  # Adjust test size based on the dataset size
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42)
    
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Unable to perform train-test split. Please check your data.")

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return clf, accuracy

# Function to plot historical closing prices
def plot_stock_prices(data, stock_name):
    st.subheader(f'{stock_name} Historical Closing Prices')
    plt.figure(figsize=(10, 5))
    plt.plot(data['Close'], label='Closing Price')
    plt.title(f'{stock_name} Historical Closing Prices')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    st.pyplot(plt)

# Function to fetch, preprocess, train model, and plot prices for Indian indices
def fetch_train_predict_indices(ticker, start_date, end_date, index_name):
    # Fetch historical data
    indices_data = fetch_indian_indices_data(ticker, start_date, end_date)

    # Preprocess the fetched data
    preprocessed_indices_data = preprocess_stock_data(indices_data)

    # Extract features and labels
    features = extract_features(preprocessed_indices_data)
    labels = preprocessed_indices_data['Label']

    # Train the model
    model, accuracy = train_model(features, labels)
    print("Model trained with accuracy:", accuracy)

    # Plot historical closing prices
    plot_stock_prices(preprocessed_indices_data, index_name)

def display_indian_index_info(index_symbol, start_date, end_date):
    st.title(f"Finance App: {index_symbol}")

    # Download historical index data
    index_data = yf.download(index_symbol, start=start_date, end=end_date)

    # Preprocess index data
    processed_data = preprocess_stock_data(index_data)

    # Features for machine learning
    features = extract_features(processed_data)

    # Train a RandomForestClassifier
    labels = processed_data['Label']
    model, accuracy = train_model(features, labels)

    # Display historical closing prices
    plot_stock_prices(processed_data, index_symbol)

    # Display model accuracy
    st.subheader(f'Model Accuracy: {accuracy:.2f}')

    # Display index information
    index_info = yf.Ticker(index_symbol)

    st.title('Index News Analysis')

    # Fetch news articles
    articles = fetch_news(index_symbol)
    
    if articles:
        # Display news articles
        st.subheader('Latest News Articles')
        for article in articles:
            st.write(f"**{article['title']}**")
            st.write(article['description'])
            st.write(f"Source: {article['source']['name']}")
            st.write(f"Published At: {article['publishedAt']}")
            st.write("---")

        # Perform sentiment analysis
        sentiment_scores = analyze_news_sentiment(articles)

        # Calculate average sentiment score
        avg_sentiment = np.mean(sentiment_scores)

        # Display sentiment analysis
        st.subheader('News Sentiment Analysis')
        st.write(f"Average Sentiment Score: {avg_sentiment:.2f}")

        # Visualize sentiment scores
        sentiment_df = pd.DataFrame({'Sentiment Score': sentiment_scores})
        st.line_chart(sentiment_df)

    else:
        st.write('No news articles found for the specified index symbol.')
    
    # Make predictions on new data
    new_data = features.iloc[-1, :].values.reshape(1, -1)
    prediction = model.predict(new_data)
    st.subheader(f'Predicted Label for {index_symbol}: {prediction[0]}')

    # Generate investment recommendation
    if prediction[0] == 1:
        st.success("The model predicts that the index value will go up. Consider investing.")
    else:
        st.warning("The model predicts that the index value will go down. Consider caution.")



def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title('Finance App')
    with st.sidebar:
        # st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
        st.sidebar.image("image-650x150.jpg")
        st.title("FINTORR PRIVATE LIMITED")
        st.write("Our finance bot, accessible via Palm chat, offers stock predictions using machine learning and sentiment analysis, ideal for novice investors. Users can query predictions, manage their portfolio, and utilize a currency converter. Simplifying stock market complexities")
        # Sidebar options
    selected_option = st.sidebar.selectbox('Select Action', ('Market Data', 'Trading Strategy', 'Portfolio', 'Indian Indices Real-time Data'))

    models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
    selected_model = st.sidebar.selectbox("Select Chatbot Model", models, format_func=lambda x: x.name)

    if selected_option == 'Market Data':
        st.sidebar.title('Market Data')
        # Get user input for stock symbol and date range
        symbol = st.sidebar.text_input('Enter Stock Symbol', 'AAPL')
        start_date = st.sidebar.date_input('Start Date', datetime.date(2020, 1, 1))
        end_date = st.sidebar.date_input('End Date', datetime.date.today())

        if start_date < end_date:
            st.write(f"Fetching historical data for {symbol} from {start_date} to {end_date}...")
            data = fetch_historical_data(symbol, start_date, end_date)
            if data:
                df = pd.DataFrame(data)
                st.write(df)

                st.write("V: Volume (number of shares traded)")

                # Display volume weighted average price
                st.write("VW: Volume Weighted Average Price (average price weighted by trading volume)")

                # Display open price
                st.write("O: Open price (price at the start of the trading period)")

                # Display high price
                st.write("H: High price (highest price during the trading period)")

                # Display low price
                st.write("L: Low price (lowest price during the trading period)")

                # Display close price
                st.write("T: Close price (price at the end of the trading period)")


                # Display date
                st.write("N: Date (timestamp indicating the trading date)")

                # Plot historical data
                plt.figure(figsize=(12, 6))
                plt.plot(df['t'], df['c'], label='Close Price', color='blue')
                plt.title('Historical Stock Prices')
                plt.xlabel('Date')
                plt.ylabel('Price')
                plt.legend()
                st.pyplot()
            else:
                st.write('No data available for the specified symbol and date range.')

        # Add Palm API prompt
        st.header("Advanced Chatbot Initialization B")
        prompt = st.text_input("Ask your question based on the market data:", "")
        if st.button("Generate Response"):
            response = generate_text(prompt, model=selected_model.name)
            st.subheader("Advanced Chatbot Initialization B Response:")
            st.write(response)
        
        if st.button("Predict Stock movement and Generate Information About Stock"):
            display_stock_info(symbol, selected_model)

    elif selected_option == 'Trading Strategy':
        st.sidebar.title('Trading Strategy')
        # Get user input for stock symbol and date range
        symbol = st.sidebar.text_input('Enter Stock Symbol', 'AAPL')
        start_date = st.sidebar.date_input('Start Date', datetime.date(2020, 1, 1))
        end_date = st.sidebar.date_input('End Date', datetime.date.today())

        if start_date < end_date:
            st.write(f"Fetching historical data for {symbol} from {start_date} to {end_date}...")
            data = fetch_historical_data(symbol, start_date, end_date)
            if data:
                df = pd.DataFrame(data, columns=['t', 'o', 'h', 'l', 'c', 'v'])
                df['t'] = pd.to_datetime(df['t'], unit='ms')
                df.set_index('t', inplace=True)

                # Simulate trading strategy
                df = simulate_trading_strategy(df)

                # Plot closing prices and moving averages
                st.write('Closing Prices and Moving Averages')
                fig, ax1 = plt.subplots(figsize=(12, 6))

                # Plot stock price and moving averages
                ax1.plot(df.index, df['c'], label='Close Price', color='blue')
                ax1.plot(df.index, df['MA50'], label='50-Day MA', color='orange')
                ax1.plot(df.index, df['MA200'], label='200-Day MA', color='red')

                # Plot buy and sell signals
                buy_signals = df.index[df['Signal'] == 1]
                sell_signals = df.index[df['Signal'] == 0]

                if len(buy_signals) > 0:
                    ax1.scatter(buy_signals, df['c'][df['Signal'] == 1], marker='^', color='green', label='Buy Signal', lw=2)

                if len(sell_signals) > 0:
                    ax1.scatter(sell_signals, df['c'][df['Signal'] == 0], marker='v', color='red', label='Sell Signal', lw=2)

                ax1.set_xlabel('Date')
                ax1.set_ylabel('Price')
                ax1.legend(loc='upper left')

                plt.title('Stock Price and Moving Averages')

                st.pyplot(fig)

                ax1.set_xlabel('Date')
                ax1.set_ylabel('Price')
                ax1.legend(loc='upper left')
                plt.title('Stock Price and Moving Averages')

                # Smulate trading strategy
                df = simulate_trading_strategy(df)
                # Plot moving averages
                fig_ma = plot_moving_averages(df)
                st.pyplot(fig_ma)
                # Calculate and plot RSI
                df = calculate_rsi(df)
                fig_rsi = plot_rsi(df)
                st.pyplot(fig_rsi)
                # Calculate and plot Bollinger Bands
                # df = calculate_bollinger_bands(df)
                # fig_bb = plot_bollinger_bands(df)
                # st.pyplot(fig_bb)
                # Calculate and plot MACD
                df = calculate_macd(df)
                fig_macd = plot_macd(df)
                st.pyplot(fig_macd)

        # Add Palm API prompt
        st.header("Advanced Chatbot Initialization B")
        prompt = st.text_input("Ask your question based on the trading strategy data:", "")
        if st.button("Generate Response"):
            response = generate_text(prompt, model=selected_model.name)
            st.subheader("Advanced Chatbot Initialization B Response:")
            st.write(response)

    elif selected_option == 'Portfolio':
        st.sidebar.title('Portfolio')

        # Get user input for stock symbol and date range
        symbol = st.sidebar.text_input('Enter Stock Symbol', 'AAPL')
        start_date = st.sidebar.date_input('Start Date', datetime.date(2020, 1, 1))
        end_date = st.sidebar.date_input('End Date', datetime.date.today())


        st.subheader('Dynamic Portfolio Value Over Time')
        # Initial investment
        initial_investment = st.sidebar.number_input('Initial Investment', min_value=0, value=10000)

        # Quantity to buy and sell
        buy_quantity = st.sidebar.number_input('Buy Quantity', min_value=0, value=0)
        sell_quantity = st.sidebar.number_input('Sell Quantity', min_value=0, value=0)

        if start_date < end_date:
            st.write(f"Fetching historical data for {symbol} from {start_date} to {end_date}...")
            data = fetch_historical_data(symbol, start_date, end_date)
            if data:
                df = pd.DataFrame(data)
                st.write(df)

                
                st.title('Currency Conversion')
                amount = st.number_input("Enter the amount to convert:", min_value=0.01, step=0.01)  # Amount in base currency
                from_currency = st.text_input("Enter the source currency (e.g., USD):").upper()
                to_currency = st.text_input("Enter the target currency (e.g., EUR):").upper()

                if st.button('Convert'):
                    # Fetch exchange rate data
                    exchange_data = fetch_exchange_rate(from_currency, to_currency)

                    # Perform currency conversion
                    converted_amount = convert_currency(amount, exchange_data)

                    # Plot exchange rate
                    plot_exchange_rate(exchange_data, from_currency, to_currency)

                    # Display the converted amount
                    st.write(f'{amount} {from_currency} is equivalent to {converted_amount.iloc[-1]:.2f} {to_currency}')

            else:
                st.write('No data available for the specified symbol and date range.')

            # Calculate portfolio value
            portfolio_value = calculate_portfolio_value(df, initial_investment, buy_quantity, sell_quantity)

            # Plot dynamic portfolio value over time
            if 'df' in locals():
                portfolio_values = [initial_investment + (buy_quantity - sell_quantity) * df['c'].iloc[i] for i in range(len(df))]
                plt.figure(figsize=(12, 6))
                plt.plot(df.index, portfolio_values, label='Portfolio Value', color='green')
                plt.title('Dynamic Portfolio Value Over Time')
                plt.xlabel('Date')
                plt.ylabel('Portfolio Value')
                plt.legend()
                st.pyplot()


        
        
        portfolio_ticker = "AAPL"  # Example portfolio holding (Apple)
        benchmark_ticker = '^GSPC'  # S&P 500 index
        portfolio_prices = fetch_historical_data1(portfolio_ticker, start_date, end_date)
        benchmark_prices = fetch_historical_data1(benchmark_ticker, start_date, end_date)

        # Calculate daily returns
        portfolio_returns = portfolio_prices.pct_change().dropna()
        benchmark_returns = benchmark_prices.pct_change().dropna()

        # Calculate cumulative returns
        portfolio_cumulative_returns = (1 + portfolio_returns).cumprod() - 1
        benchmark_cumulative_returns = (1 + benchmark_returns).cumprod() - 1
        if st.button("Difference btw S&P 500 and Portfolio "):
            plt.figure(figsize=(10, 6))
            plt.plot(portfolio_cumulative_returns, label='Portfolio', color='blue')
            plt.plot(benchmark_cumulative_returns, label='S&P 500', color='red')
            plt.title('Portfolio vs. S&P 500 Performance')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Return')
            plt.legend()
            st.pyplot()

            data = fetch_historical_data1(symbol, start_date, end_date)
            rsi = calculate_rsi1(data)

            threshold_low = 30
            threshold_high = 70

            # Generate notifications
            notifications = generate_notifications(rsi, threshold_low, threshold_high)

            # Plot stock price and RSI with notifications
            plt.figure(figsize=(12, 6))
            plt.plot(data.index, data['Close'], label='Stock Price', color='blue')
            plt.plot(data.index, rsi, label='RSI', color='orange')
            for idx, notification in enumerate(notifications):
                plt.text(data.index[idx], data['Close'].iloc[idx], notification, fontsize=8, color='red')
            plt.title('Stock Price and RSI with Risk Notifications')
            plt.xlabel('Date')
            plt.ylabel('Price / RSI')
            plt.legend()
            # plt.show()
            st.pyplot()



    
       

        # Add Palm API prompt
        st.header("Advanced Chatbot Initialization B")
        prompt = st.text_input("Ask your question based on the portfolio data:", "")
        if st.button("Generate Response"):
            response = generate_text(prompt, model=selected_model.name)
            st.subheader("Advanced Chatbot Initialization B Response:")
            st.write(response)

    elif selected_option == 'Indian Indices Real-time Data':
        with st.sidebar:
            st.sidebar.title('Select Index')
            symbol = st.sidebar.radio('Select Index', ('^NSEI', '^NSEBANK', '^BSESN'))
            start_date = st.sidebar.date_input('Start Date', datetime.date(2020, 1, 1))
            end_date = st.sidebar.date_input('End Date', datetime.date.today())

        # if st.button('Fetch Real-time Data'):
        #     # Fetch real-time data
        data = fetch_indian_indices_data(symbol, start_date, end_date)
        # Plot real-time data
        st.subheader('Real-time Data')
        st.write(data)
        data = fetch_realtime_data(symbol)
        data1 = fetch_historical_data2(symbol)
        # Plot real-time data
        plot_realtime_data(data, symbol)
        # Plot historical data
        plot_historical_line_chart(data1)
        # Plot candlestick chart
        plot_candlestick_chart(data)

        if st.button('Predict'):
            # Call function to display index information
            display_indian_index_info(symbol, start_date, end_date)
        st.header("Advanced Chatbot Initialization B")
        prompt = st.text_input("Ask your question based on the portfolio data:", "")
        if st.button("Generate Response"):
            response = generate_text(prompt, model=selected_model.name)
            st.subheader("Advanced Chatbot Initialization B Response:")
            st.write(response)





if __name__ == '__main__':
    main()
