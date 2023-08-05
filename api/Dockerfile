FROM python:3.10-slim-buster
WORKDIR /api
COPY . ./
RUN pip install -r requirements.txt
EXPOSE 3000
CMD ["python","main.py"]