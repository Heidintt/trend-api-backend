# trend_fetcher.py

import pandas as pd
from pytrends.request import TrendReq
from prophet import Prophet

def fetch_google_trends(keywords, timeframe='today 3-m'):
    """
    Lấy dữ liệu Google Trends với cơ chế xử lý lỗi.
    """
    try:
        pytrends = TrendReq(hl='en-US', tz=360)
        pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo='', gprop='')
        data = pytrends.interest_over_time()
        if data.empty:
            return pd.DataFrame()
        if 'isPartial' in data.columns:
            data = data.drop(columns=['isPartial'])
        return data.reset_index()
    except Exception as e:
        print(f"Lỗi khi lấy dữ liệu Google Trends: {e}")
        return pd.DataFrame()

def detect_spikes(series, window=7, threshold=2.0):
    """
    Phát hiện các điểm tăng đột biến trong một chuỗi dữ liệu.
    """
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std().fillna(0)
    z_score = ((series - rolling_mean) / (rolling_std + 1e-9)).fillna(0)
    return z_score > threshold

def generate_forecast(df, keyword):
    """
    Tạo dữ liệu dự báo cho 90 ngày tới bằng Prophet.
    """
    # Chuẩn bị dữ liệu cho Prophet (yêu cầu cột 'ds' và 'y')
    prophet_df = df[['date', keyword]].rename(columns={'date': 'ds', keyword: 'y'})
    
    # Khởi tạo và huấn luyện mô hình
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(prophet_df)
    
    # Tạo DataFrame cho tương lai và dự báo
    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)
    
    # Chỉ trả về các cột cần thiết của phần dự báo
    return forecast.tail(90)[['ds', 'yhat']]

def analyze_trend_data(keyword, timeframe_str):
    """
    Hàm chính để điều phối tất cả các bước: Lấy dữ liệu, phát hiện spike, dự báo.
    """
    # Ánh xạ timeframe từ frontend sang định dạng của pytrends
    timeframe_mapping = {
        "7d": "now 7-d",
        "30d": "today 1-m",
        "3m": "today 3-m",
        "12m": "today 12-m",
    }
    pytrends_timeframe = timeframe_mapping.get(timeframe_str)
    if not pytrends_timeframe:
        return None # Trả về None nếu timeframe không hợp lệ

    # 1. Lấy dữ liệu lịch sử
    historical_df = fetch_google_trends([keyword], timeframe=pytrends_timeframe)
    if historical_df.empty:
        return None

    # 2. Phát hiện đột biến (Spikes)
    historical_df['is_spike'] = detect_spikes(historical_df[keyword])
    
    # 3. Tạo dữ liệu dự báo (Forecast)
    prediction_df = generate_forecast(historical_df, keyword)

    # 4. Định dạng lại dữ liệu để trả về dưới dạng JSON chuẩn
    historical_data = [
        {
            "date": row['date'].strftime('%Y-%m-%d'),
            "value": row[keyword],
            "is_spike": bool(row['is_spike'])
        }
        for index, row in historical_df.iterrows()
    ]

    prediction_data = [
        {
            "date": row['ds'].strftime('%Y-%m-%d'),
            "predicted_value": round(row['yhat'])
        }
        for index, row in prediction_df.iterrows()
    ]

    return {
        "historical": historical_data,
        "prediction": prediction_data
    }
