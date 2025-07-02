# main.py

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from trend_fetcher import analyze_trend_data # Import hàm xử lý chính

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="Market Trend Analysis API",
    description="An API to fetch, analyze, and predict Google Trends data.",
    version="1.0.0"
)

# Cấu hình CORS để cho phép frontend gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả các domain
    allow_credentials=True,
    allow_methods=["GET"], # Chỉ cho phép phương thức GET
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Welcome to the Trend Analysis API!"}

# API endpoint chính để phân tích xu hướng
@app.get("/api/analyze")
async def analyze_trends(
    keyword: str = Query(..., description="The keyword to analyze, e.g., 'AI'"),
    timeframe: str = Query(..., description="Timeframe for analysis: '7d', '30d', '3m', '12m'")
):
    """
    Phân tích xu hướng cho một từ khóa: lấy dữ liệu lịch sử, phát hiện đột biến, và dự báo.
    """
    if not keyword or not timeframe:
        raise HTTPException(
            status_code=400, 
            detail="Both 'keyword' and 'timeframe' parameters are required."
        )

    try:
        # Gọi hàm xử lý chính từ trend_fetcher
        analysis_result = analyze_trend_data(keyword, timeframe)

        if not analysis_result:
            raise HTTPException(
                status_code=404, 
                detail=f"No data found for the keyword '{keyword}' or invalid timeframe."
            )
        
        return analysis_result
        
    except Exception as e:
        # Bắt các lỗi không mong muốn
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(
            status_code=500, 
            detail="An internal server error occurred."
        )
