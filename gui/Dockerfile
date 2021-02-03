FROM python

WORKDIR /gui 

COPY gui_app.py .
COPY gunicorn_starter.sh .
COPY requirements.txt .
COPY templates/ ./templates/

RUN pip install -r requirements.txt\
  && chmod +x gunicorn_starter.sh

CMD [ "./gunicorn_starter.sh" ]