'''
from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, Azure!"

if __name__ == '__main__':
    app.run(debug=True)
'''


from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI on Azure!"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

