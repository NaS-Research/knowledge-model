"""
main.py
Minimal FastAPI app so our Render service has a running server.
"""

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Knowledge Model API is alive!"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
