from pydantic import BaseModel, Field, field_validator
from typing import Optional

class CustomerInput(BaseModel):
    customerID: Optional[str] = None
    gender: str = Field(..., description="Male or Female")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="0 or 1")
    Partner: str = Field(..., description="Yes or No")
    Dependents: str = Field(..., description="Yes or No")
    tenure: int = Field(..., ge=0, description="Months with company")
    PhoneService: str = Field(..., description="Yes or No")
    MultipleLines: str = Field(..., description="Yes, No, or No phone service")
    InternetService: str = Field(..., description="DSL, Fiber optic, or No")
    OnlineSecurity: str = Field(..., description="Yes, No, or No internet service")
    OnlineBackup: str = Field(..., description="Yes, No, or No internet service")
    DeviceProtection: str = Field(..., description="Yes, No, or No internet service")
    TechSupport: str = Field(..., description="Yes, No, or No internet service")
    StreamingTV: str = Field(..., description="Yes, No, or No internet service")
    StreamingMovies: str = Field(..., description="Yes, No, or No internet service")
    Contract: str = Field(..., description="Month-to-month, One year, or Two year")
    PaperlessBilling: str = Field(..., description="Yes or No")
    PaymentMethod: str = Field(..., description="Electronic check, Mailed check, Bank transfer (automatic), or Credit card (automatic)")
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)

    # ================== VALIDATION ==================
    @field_validator('gender')
    @classmethod
    def validate_gender(cls, v):
        if v not in ['Male', 'Female']:
            raise ValueError('gender must be Male or Female')
        return v

    @field_validator('Partner', 'Dependents', 'PhoneService', 'PaperlessBilling')
    @classmethod
    def validate_yes_no(cls, v):
        if v not in ['Yes', 'No']:
            raise ValueError('Value must be Yes or No')
        return v

    @field_validator('MultipleLines')
    @classmethod
    def validate_multiple_lines(cls, v):
        valid = ['Yes', 'No', 'No phone service']
        if v not in valid:
            raise ValueError(f'Must be one of: {valid}')
        return v

    @field_validator('InternetService')
    @classmethod
    def validate_internet(cls, v):
        valid = ['DSL', 'Fiber optic', 'No']
        if v not in valid:
            raise ValueError(f'Must be one of: {valid}')
        return v

    @field_validator('OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies')
    @classmethod
    def validate_internet_services(cls, v):
        valid = ['Yes', 'No', 'No internet service']
        if v not in valid:
            raise ValueError(f'Must be one of: {valid}')
        return v

    @field_validator('Contract')
    @classmethod
    def validate_contract(cls, v):
        valid = ['Month-to-month', 'One year', 'Two year']
        if v not in valid:
            raise ValueError(f'Must be one of: {valid}')
        return v

    @field_validator('PaymentMethod')
    @classmethod
    def validate_payment(cls, v):
        valid = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
        if v not in valid:
            raise ValueError(f'Must be one of: {valid}')
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "customerID": "7590-VHVEG",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85
            }
        }


class PredictionOutput(BaseModel):
    customer_id: Optional[str] = None
    churn_prediction: int
    churn_probability: float
    risk_level: str
    message: str
    

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool