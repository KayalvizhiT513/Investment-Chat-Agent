from fastapi import FastAPI
from app.routes import router

app = FastAPI(
    title="Performance Analytics Formula API",
    version="1.0.0",
    description="Computes key investment performance metrics."
)

app.include_router(router)



@app.get("/")
def root():
    return {"message": "✅ Formula API is live.", "endpoints": ["/metrics", "/compute"]}
