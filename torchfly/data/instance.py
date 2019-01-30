from abc import ABC, abstractmethod

class Instance(ABC):
    def __init__(self, index):
        self.index = index
        super().__init__()
    
    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass