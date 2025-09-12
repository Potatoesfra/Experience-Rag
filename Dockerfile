FROM python:3.10.18-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
COPY . .

CMD ["python", "Rag_Lucas.py"]
