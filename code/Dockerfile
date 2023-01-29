FROM python:3
WORKDIR /code
COPY . .
COPY ./requirements.txt .

ENV FLASK_APP=main.py
RUN pip install --no-cache-dir -r requirements.txt
# By default listen on port 5000
EXPOSE 5000/tcp
RUN export FLASK_APP=main.py
CMD ["python", "main.py"]
