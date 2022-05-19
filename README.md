This is a fork of [knowledgestream](https://github.com/shiralkarprashant/knowledgestream) originally created by [shiralkarprashant](https://github.com/shiralkarprashant).
This fork introduces a TCP interface that can be used to submit assertions to the different fact validation approaches, instead of reading the assertions from a file.

Also added a docker container for easy execution.

# Paper
Originally presented in the paper "Finding Streams in Knowledge Graphs to Support Fact Checking" in proceedings of ICDM 2017. A full version of this paper can be found at: https://arxiv.org/abs/1708.07239

# Fetch
```
git clone https://github.com/saschaTrippel/knowledgestream
cd knowledgestream
```

# Data
```
curl --output data.zip https://zenodo.org/record/4487154/files/data.zip

```

# System requirements

* Docker

# Install from source

```python setup.py build_ext -if```

```python setup.py install```

Note: for the second command, please do sudo in case you need installation rights on the machine.

### Requires:

* cython
* numpy
* scipy
* pandas
* ujson
* sklearn
* rdflib

# Install using docker
Execute in knowledgestream directory.
```
docker build -t knowledgestream .
```

# Run source
```
kstream -m stream -p 4444
```
* **m:** Method to use.
* **p:** Port on which knowledgestream will listen for assertions.

# Run docker
```
docker run -p 127.0.0.1:4444:4444 knowledgestream
```

# Usage
By default, the container listens on port 4444 for assertions in turtle format.

The response to an assertion is the veracity score of that assertion or PARSING ERROR in case the assertion is not in turtle format.

# Methods
The available fact validation methods and their respective arguments are:

* Knowledgestream [stream]
* REL-KL [relklinker]
* KL [klinker]
* Katz [katz]
* Pathent [pathent]
* SimRank [simrank]
* Adamic Adar [adamic_adar]
* Jaccard [jaccard]
* Degree Product [degree_product]