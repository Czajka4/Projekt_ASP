FROM python

WORKDIR /api
COPY api_app.py .
COPY gunicorn_starter.sh .
RUN pip install flask flask-cors json_logging gunicorn \
  && chmod +x gunicorn_starter.sh

CMD [ "./gunicorn_starter.sh" ]