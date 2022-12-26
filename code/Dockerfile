FROM python:3
COPY . .
WORKDIR /code
COPY ./requirements.txt .

ENV FLASK_APP=main.py
RUN pip install --no-cache-dir -r requirements.txt
ENV FLASK_APP=main.py
CMD ["flask", "--app", "main", "run"]