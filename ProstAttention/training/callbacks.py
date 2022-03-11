from keras.callbacks import Callback


class LossWeightsCallback(Callback):
    def __init__(self, output_branch, update_epoch: int = 20, target_weight: float = 1.):
        """ Keras Callback to update loss weights at a given epoch.

        Args:
            output_branch: str, name of the model's output branch where loss should be changed
            update_epoch: int, epoch on which the weights on losses should be changed
            target_weight: float, value that the loss_weight should be changed to
        """
        super().__init__()
        self.output_branch = output_branch
        self.update_epoch = update_epoch
        self.target_weight = target_weight

    def on_epoch_end(self, epoch, logs=None):
        """ Method called at the end of each epoch.
            Updates the loss weights when a specified epoch is reached.

        Args:
            epoch: int, current epoch
            logs: None
        """
        if epoch <= self.update_epoch:
            # Update the loss weight associated to the specified model's output branch
            # and re-compile the model
            if epoch == self.update_epoch:
                self.model.loss_weights[self.output_branch] = self.target_weight
                self.model.compile(
                    optimizer=self.model.optimizer,
                    loss=self.model.loss,
                    metrics=self.model.metrics,
                    loss_weights=self.model.loss_weights
                )
            print(f"epoch {epoch + 1}, weights = {self.model.loss_weights}")
