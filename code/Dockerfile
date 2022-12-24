FROM python:3
COPY . .
WORKDIR /code
COPY ./requirements.txt .

ENV FLASK_APP=main.py
RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]