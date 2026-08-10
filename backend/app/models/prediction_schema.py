from pydantic import BaseModel


class PredictionRequest(BaseModel):
    Duration: int
    Date: int
    Pulse: int
    Maxpulse: int