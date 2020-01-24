from python:3.7

WORKDIR /tmp/build

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

# Install fastFM.
# For OpenBLAS to work on Ryzen 7, we had to replace OpenBLAS with newer version.
RUN git clone --recursive https://github.com/ibayer/fastFM.git && \
    cd fastFM/fastFM-core/externals && rm -rf OpenBlas && \
    wget https://github.com/xianyi/OpenBLAS/archive/v0.3.7.zip && \
    unzip v0.3.7.zip && \
    mv OpenBLAS-0.3.7 OpenBLAS && \
    cd /tmp/build/fastFM && pip install -r requirements.txt && make && \
    pip install .

WORKDIR /

# Downloading benchmark data...
RUN wget http://files.grouplens.org/datasets/movielens/ml-100k.zip && \
    mv ml-100k.zip /.ml-100k.zip


COPY experiments/ /experiments/
COPY run_experiments_all.sh /run_experiments_all.sh
CMD ["./run_experiments_all.sh"]
