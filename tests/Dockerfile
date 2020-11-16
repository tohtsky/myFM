FROM python:3.6

RUN pip install --index-url https://test.pypi.org/simple --extra-index-url https://pypi.org/simple myFM==0.3.0.0.dev1
RUN pip install scikit-learn pandas 
WORKDIR /work/
RUN wget http://files.grouplens.org/datasets/movielens/ml-100k.zip -O /root/.ml-100k.zip
ADD oprobit.py regressions.py /work/

ENTRYPOINT ["python", "regressions.py"]