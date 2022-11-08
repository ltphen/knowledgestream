import logging as log
import socket
from datastructures.Message import Message
from algorithms.AlgorithmRunner import AlgorithmRunner

STATES = ["waiting", "training", "testing"]

class ClientService:

    def __init__(self, client, method, graph, relsim, internalId):
        self._state = "waiting"
        self.type = self._type(method)
        self.client = client
        self.algoRunner = AlgorithmRunner(method, graph, internalId, relsim)

    def serve(self):
        while True:
            try:
                log.info('Waiting for a request')
                request = self.client.recv(1024)
                if request == '':
                    log.info('Connection closed')
                    self.client.close()
                    return

                log.info('### REQEST RECEIVED ###')
                requestMessage = self.parseRequest(request)
                response = self.respondToRequest(requestMessage)
                log.info('### REQUEST ANSWERED ###')
                self.client.send(response.serialize())

            except socket.error as ex:
                log.info('Socket error occured.')
                return
            except KeyError as ex:
                self.client.send(Message(type="error", content="ID Error").serialize())
                continue
            except UnicodeEncodeError as ex:
                self.client.send(Message(type="error", content="Encoding Error").serialize())
                continue
            except Exception as ex:
                raise ex
    
    def parseRequest(self, assertionString):
        log.info('Parsing request: {}'.format(assertionString.replace('\n', '')))
        return Message(text=assertionString)
    
    def respondToRequest(self, request):
        if request.type == "call" and request.content == "type":
            return Message(type="type_response", content=self.type)

        if self.state == "waiting" and request.type == "call" and request.content == "training_start":
            self.state = "training"
            return Message(type="ack", content="training_start_ack")

        if self.state == "training" and request.type == "train":
            self.algoRunner.addTrainingData(request.subject, request.predicate, request.object, request.score)
            return Message(type="ack", content="train_ack")

        if self.state == "training" and request.type == "call" and request.content == "training_complete":
            self.algoRunner.train()
            self.state = "testing"
            return Message(type="ack", content="training_complete_ack")
    
        if self.state == "testing" and request.type == "test":
            result = self.algoRunner.test(request.subject, request.predicate, request.object)
            return Message(type="test_result", score="{:f}".format(result))
    
        return Message(type="error", content="Something went wrong.")


    def _type(self, method):
        if method in ["predpath", "pra"]:
            return "supervised"
        else:
            self.state = "testing"
            return "unsupervised"

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        if state in STATES:
            self._state = state

