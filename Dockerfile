FROM debian:11 AS knowledgestream

# Get required software packages
RUN apt-get update
RUN apt-get install -y build-essential unzip curl python2 python2-dev git

# Get required python2 modules
RUN curl --output get-pip.py https://bootstrap.pypa.io/pip/2.7/get-pip.py
RUN python2 get-pip.py

RUN python2 -m pip install cython
RUN python2 -m pip install numpy
RUN python2 -m pip install scipy
RUN python2 -m pip install pandas
RUN python2 -m pip install ujson
RUN python2 -m pip install sklearn
RUN python2 -m pip install rdflib

# Get knowledgestream
RUN git clone 'https://github.com/saschaTrippel/knowledgestream.git'
WORKDIR /knowledgestream
RUN mkdir output

# Install knowledgestream
RUN python2 setup.py build_ext -if
RUN python2 setup.py install

# Get knowledge graph from zenodo.org
RUN curl --output data.zip https://zenodo.org/record/4487154/files/data.zip

# Unpack knowledgegraph
RUN unzip data.zip

# Expose port
EXPOSE 4444/tcp

# Start knowledgestream
CMD kstream -m stream -p 4444
