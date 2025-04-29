import os
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from predict import PortfolioOptimizer
from typing import Dict, List, Optional
from pydantic import BaseModel
import json
from datetime import datetime

# Create FastAPI app
app = FastAPI(
    title="Portfolio Optimization API",
    description="API for optimal portfolio allocation",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create outputs directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Initialize portfolio optimizer
optimizer = PortfolioOptimizer()


class PerformanceResponse(BaseModel):
    cumulative_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    evaluation_date: str
    evaluation_period_days: int


class AllocationResponse(BaseModel):
    allocation: Dict[str, float]
    model_training_date: str


class RecommendationResponse(BaseModel):
    optimal_allocation: Dict[str, float]
    performance_metrics: PerformanceResponse
    recommendation_date: str
    model_training_date: str


@app.get("/", response_model=Dict[str, str])
def read_root():
    """Get API information."""
    return {
        "name": "Portfolio Optimization API",
        "version": "1.0.0",
        "description": "API for optimal portfolio allocation"
    }


@app.get("/allocation", response_model=AllocationResponse)
def get_allocation():
    """Get the current optimal portfolio allocation."""
    allocation = optimizer.get_portfolio_allocation()
    return {
        "allocation": allocation,
        "model_training_date": optimizer.model["training_date"]
    }


@app.get("/performance", response_model=PerformanceResponse)
def get_performance(days: int = Query(30, description="Number of days for performance evaluation")):
    """Calculate portfolio performance for the specified number of days."""
    try:
        metrics, _ = optimizer.calculate_portfolio_performance(days=days)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating performance: {str(e)}")


@app.get("/recommendation", response_model=RecommendationResponse)
def get_recommendation(days: int = Query(30, description="Number of days for performance evaluation")):
    """Generate investment recommendation based on the optimal portfolio."""
    try:
        # Force recalculation with the specified number of days
        metrics, _ = optimizer.calculate_portfolio_performance(days=days)
        
        # Get allocation
        allocation = optimizer.get_portfolio_allocation()
        sorted_allocation = {k: v for k, v in sorted(allocation.items(), key=lambda item: item[1], reverse=True)}
        
        # Generate recommendation
        recommendation = {
            "optimal_allocation": sorted_allocation,
            "performance_metrics": metrics,
            "recommendation_date": datetime.now().strftime('%Y-%m-%d'),
            "model_training_date": optimizer.model["training_date"],
        }
        
        # Save recommendation
        with open("outputs/recommendation.json", "w") as f:
            json.dump(recommendation, f, indent=4)
        
        return recommendation
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendation: {str(e)}")


@app.get("/chart", response_class=FileResponse)
def get_chart(days: int = Query(30, description="Number of days for performance chart")):
    """Generate and return portfolio performance chart."""
    try:
        # Generate chart
        optimizer.calculate_portfolio_performance(days=days)
        chart_path = "outputs/portfolio_performance.png"
        
        if not os.path.exists(chart_path):
            raise HTTPException(status_code=404, detail="Chart not found")
        
        return FileResponse(chart_path)
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error generating chart: {str(e)}")


if __name__ == "__main__":
    # Run FastAPI server
    uvicorn.run("api:app", host="0.0.0.0", port=5161, reload=True) 