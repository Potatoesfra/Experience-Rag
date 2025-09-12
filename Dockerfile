FROM python:3.10.18-slim
WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir torch==2.3.1+cu118 --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir faiss-gpu --extra-index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "Rag_Lucas.py"]
