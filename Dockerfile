FROM python:3
WORKDIR /bdml

COPY run.sh .
COPY requirements.txt .

RUN pip install -r requirements.txt

COPY model.pickle .
COPY text_vectorizer.pickle .
COPY predict.py .

CMD ["bash", "run.sh"]