# ใช้ base image ที่มี Python
FROM python:3.9-slim

# ติดตั้ง dependencies ที่จำเป็น
RUN apt-get update && \
    apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ติดตั้ง Python packages
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกโค้ดเข้า container
COPY . .

# รันโปรแกรมหลัก
CMD ["python", "main_edit.py"]

# docker build -t ven_robot