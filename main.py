from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from app.schemas import CustomerInput, PredictionOutput, HealthResponse
from app.predict import predict_churn, model
from typing import List

# ================== CREATE APP ==================
app = FastAPI(
    title="Churn Prediction API",
    description="Predict customer churn using XGBoost model",
    version="1.0.0"
)

# ================== CUSTOM ERROR HANDLER ==================
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    errors = []
    for error in exc.errors():
        errors.append({
            "field": error["loc"][-1],
            "message": error["msg"]
        })
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "message": "Validation failed",
            "errors": errors
        }
    )

# ================== ENDPOINTS ==================

# Home
@app.get("/")
def home():
    return {
        "message": "🚀 Churn Prediction API is running!",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "batch": "/predict/batch (POST)",
            "docs": "/docs"
        }
    }

# Health Check
@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

# Single Prediction
@app.post("/predict", response_model=PredictionOutput)
def predict(customer: CustomerInput):
    try:
        data = customer.dict()
        result = predict_churn(data)
        return result
    
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing feature: {str(e)}"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

# Batch Prediction
@app.post("/predict/batch")
def predict_batch(customers: List[CustomerInput]):
    try:
        results = []
        for customer in customers:
            data = customer.dict()
            result = predict_churn(data)
            results.append(result)
        
        return {
            "status": "success",
            "count": len(results),
            "predictions": results
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )