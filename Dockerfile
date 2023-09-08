FROM tiangolo/uwsgi-nginx-flask:python3.10

RUN groupadd -r myuser && useradd -r -g myuser myuser

USER root 

WORKDIR /streamlit-chatbot-template

COPY . .

RUN pip install -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]

CMD ["1_Home.py"]

