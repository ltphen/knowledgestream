import json

class Message:

    def __init__(self, type=None, content=None, subject=None, predicate=None, object=None, score=None, text=None):
        if text != None:
            self.parse(text)
        else:
            self._type = type
            self.content = content
            self.subject = subject
            self.predicate = predicate
            self.object = object
            self.score = score
    
    def serialize(self):
        if self.type == "test_result":
            return json.dumps({"type": self.type, "score": self.score})
        if self.type == "ack":
            return json.dumps({"type": self.type, "content": self.content})
        if self.type == "type_response":
            return json.dumps({"type": self.type, "content": self.content})
        if self.type == "error":
            return json.dumps({"type": self.type, "content": self.content})
    
    def parse(self, text):
        response = json.loads(text)
        self.type = response["type"]

        if self.type == "call":
            self.content = response["content"]
        elif self.type == "train":
            self.subject = response["subject"]
            self.predicate = response["predicate"]
            self.object = response["object"]
            self.score = response["score"]

        elif self.type == "test":
            self.subject = response["subject"]
            self.predicate = response["predicate"]
            self.object = response["object"]
            
    @property
    def type(self):
        return self._type
    
    @type.setter
    def type(self, type):
        if type in ["call", "train", "test", "test_result", "ack", "type_response", "error"]:
            self._type = type
        
        

