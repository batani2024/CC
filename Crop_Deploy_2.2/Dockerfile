# Gunakan Python 3.9 sebagai base image
FROM python:3.10-slim

# Tentukan working directory
WORKDIR /app

# Salin file requirements.txt
COPY requirements.txt /app/

# Instal dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Salin semua file ke dalam container
COPY . /app/

# Tentukan port yang digunakan aplikasi
EXPOSE 8080

# Jalankan aplikasi
CMD ["python", "main.py"]
