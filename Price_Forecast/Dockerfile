FROM python:3.9

WORKDIR /app

COPY . /app

# Install dependencies
RUN pip install -r requirements.txt && pip install openpyxl

# Pastikan file serviceAccount.json tersedia
COPY ./config/serviceAccount.json /app/config/serviceAccount.json

CMD ["python", "main.py"]