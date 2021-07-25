FROM python:3.6

RUN pip install myFM==0.3.0.1
RUN pip install scikit-learn pandas
WORKDIR /work/
RUN wget http://files.grouplens.org/datasets/movielens/ml-100k.zip -O /root/.ml-100k.zip
ADD regressions.py /work/

ENTRYPOINT ["python", "regressions.py"]
