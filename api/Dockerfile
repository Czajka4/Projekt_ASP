FROM python

WORKDIR /api

COPY api_app.py .
COPY gunicorn_starter.sh .
COPY requirements.txt .
COPY scraping.py .

RUN pip install -r requirements.txt\
  && chmod +x gunicorn_starter.sh

CMD [ "./gunicorn_starter.sh" ]