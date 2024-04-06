FROM python:3.11.5
COPY ./app.py /deploy/
COPY ./templates /deploy/templates
COPY ./requirements.txt /deploy/
COPY ./model_new_classifier.pkl /deploy/

WORKDIR /deploy/
RUN pip install -r requirements.txt
EXPOSE 80
ENTRYPOINT ["python", "app.py"]