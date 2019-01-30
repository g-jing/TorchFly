
from abc import ABC, abstractmethod

from typing import Any


class Callback(ABC):
    "Base class for callbacks that dynamically change value."

    @abstractmethod
    def on_train_begin(self):
        "To initialize constants in the callback."
        pass

    @abstractmethod
    def on_epoch_begin(self):
        "At the beginning of each epoch."
        pass

    @abstractmethod
    def on_batch_begin(self):
        "Set HP before the step is done. Returns xb, yb (which can allow us to modify the input at that step if needed)."
        pass

    @abstractmethod
    def on_loss_begin(self):
        "Called after forward pass but before loss has been computed. Returns the output (which can allow us to modify it)."
        pass

    @abstractmethod
    def on_backward_begin(self):
        """Called after the forward pass and the loss has been computed, but before backprop.
           Returns the loss (which can allow us to modify it, for instance for reg functions)"""
        pass

    @abstractmethod
    def on_backward_end(self):
        "Called after backprop but before optimizer step. Useful for true weight decay in AdamW."
        pass

    @abstractmethod
    def on_step_end(self):
        "Called after the step of the optimizer but before the gradients are zeroed."
        pass

    @abstractmethod
    def on_batch_end(self):
        "Called at the end of the batch."
        pass

    @abstractmethod
    def on_epoch_end(self):
        "Called at the end of an epoch."
        return False

    @abstractmethod
    def on_train_end(self):
        "Useful for cleaning up things and saving files/models."
        pass
