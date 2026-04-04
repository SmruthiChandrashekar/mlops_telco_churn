FROM python:3.9

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install xgboost

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]