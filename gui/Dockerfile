FROM python

WORKDIR /gui 

COPY gui_app.py .
COPY gunicorn_starter.sh .
COPY requirements.txt .
COPY templates/ ./templates/

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

RUN pip install -r requirements.txt\
  && chmod +x gunicorn_starter.sh

CMD [ "./gunicorn_starter.sh" ]