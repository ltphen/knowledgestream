# Purpose
This fork was created to be used together with [Favel](https://github.com/saschaTrippel/favel)

This is a fork of [knowledgestream](https://github.com/shiralkarprashant/knowledgestream) originally created by [shiralkarprashant](https://github.com/shiralkarprashant).

This fork introduces a TCP interface that can be used to submit assertions to the different fact validation approaches, instead of reading the assertions from a file.

# Fetch
```
git clone https://github.com/saschaTrippel/knowledgestream
cd knowledgestream
```

# Knowledgegraph
The software works with DBPedia.

The knowledgegraph has to be in a certain format.
A knowledgegraph in turtle format such as it is provided by [DBPedia](https://databus.dbpedia.org/dbpedia/collections/dbpedia-snapshot-2022-03) can be transformed using [knowledgestream-graph-transformer](https://github.com/saschaTrippel/knowledgestream-graph-transformer).

A knowledgegraph transformed from [DBPedia-2022-03](https://databus.dbpedia.org/dbpedia/mappings/mappingbased-objects/2022.03.01/mappingbased-objects_lang=en.ttl.bz2) can be downloaded [here](https://uni-paderborn.sciebo.de/s/fR2O4m7wjWGM8Dh/download).

# Install using docker
In the Dockerfile, specify which fact validation method shall be used, and on which port the container shall listen.

By default, the container runs Knowledgestream and listens on port 4444.

Execute in knowledgestream directory.
```
docker build -t knowledgestream .
```

# Run docker
```
docker run -p 127.0.0.1:4444:4444 knowledgestream
```

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
* PredPath [predpath]
* PRA [pra]