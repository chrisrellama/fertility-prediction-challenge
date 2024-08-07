FROM continuumio/anaconda3:2024.02-1

COPY environment.yml /
RUN conda env create -f /environment.yml

RUN mkdir /app
WORKDIR /app

COPY *.csv /app
COPY *.py /app
COPY *.joblib /app
COPY index.txt /app
COPY low_importance_features.txt /app
COPY index_166.txt /app

ENTRYPOINT ["conda", "run", "-n", "eyra-rank", "python", "/app/run.py"]
CMD ["predict", "/data/fake_data.csv"]
