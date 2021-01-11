from abc import ABC, abstractmethod

__all__ = (
    'BaseReplayBuffer',
)

class BaseReplayBuffer(ABC):

    @property
    @abstractmethod
    def capacity(self):
        pass

    @abstractmethod
    def push(self, transition):
        pass

    @abstractmethod
    def get(self, **kwargs):
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __bool__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass
