FROM python:3.9

WORKDIR /app

# Copy everything
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Start FastAPI server
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]