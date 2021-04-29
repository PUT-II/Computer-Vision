LABEL student1="Tomasz Kiljańczyk" student2="Michał Kalinowski"
FROM python:3.7

COPY ./src ./app/src
COPY ./requirements.txt ./app/requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/app/src/"
RUN pip install -r requirements.txt

EXPOSE 5001
ENTRYPOINT ["python", "./docker_main.py"]