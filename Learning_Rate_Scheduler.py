import tensorflow as tf
        
# One Cycle Learning Rate Scheduler function
class OneCycleLearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs, lr_max, pct_start, div_factor, final_div_factor):
        super(OneCycleLearningRateScheduler, self).__init__()
        self.total_epochs = total_epochs
        self.lr_max = lr_max
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.pct_start * self.total_epochs:
            lr = self.lr_max * (epoch / (self.pct_start * self.total_epochs))
        else:
            lr = self.lr_max * (1 + (self.final_div_factor - 1) * (self.total_epochs - epoch) / ((1 - self.pct_start) * self.total_epochs)) / self.final_div_factor
        
        # Update the learning rate of the optimizer
        optimizer = self.model.optimizer.optimizer if isinstance(self.model.optimizer, StochasticWeightAveraging) else self.model.optimizer
        tf.keras.backend.set_value(optimizer.lr, lr)
        print(f"Learning rate set to {lr} at epoch {epoch}")
        
def one_cycle_lr(total_epochs, lr_max, pct_start, div_factor, final_div_factor):
    return OneCycleLearningRateScheduler(
        total_epochs=total_epochs,
        lr_max=lr_max,
        pct_start=pct_start,
        div_factor=div_factor,
        final_div_factor=final_div_factor
    )

class StochasticWeightAveraging(tf.keras.optimizers.Optimizer):
    def __init__(self, optimizer, start_averaging=5, average_period=10, name="StochasticWeightAveraging", **kwargs):
        super(StochasticWeightAveraging, self).__init__(name, **kwargs)
        self.optimizer = optimizer
        self.start_averaging = start_averaging
        self.average_period = average_period
        self.n_averaged = 0
        self.averaged_weights = None

    def apply_gradients(self, grads_and_vars, name=None, experimental_aggregate_gradients=True):
        return self.optimizer.apply_gradients(grads_and_vars, name, experimental_aggregate_gradients)

    def _create_slots(self, var_list):
        self.optimizer._create_slots(var_list)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        return self.optimizer._resource_apply_dense(grad, var, apply_state)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        return self.optimizer._resource_apply_sparse(grad, var, indices, apply_state)

    def average_weights(self, model):
        if self.n_averaged == 0:
            self.averaged_weights = [tf.Variable(w, trainable=False) for w in model.weights]
        else:
            for i, w in enumerate(model.weights):
                self.averaged_weights[i].assign((self.averaged_weights[i] * self.n_averaged + w) / (self.n_averaged + 1))
        self.n_averaged += 1

    def assign_average_weights(self, model):
        if self.averaged_weights is not None:
            for i, w in enumerate(self.averaged_weights):
                model.weights[i].assign(w)

    def on_epoch_end(self, epoch, logs=None, model=None):
        if epoch >= self.start_averaging and (epoch - self.start_averaging) % self.average_period == 0:
            self.average_weights(model)

    def on_train_end(self, logs=None, model=None):
        self.assign_average_weights(model)

    def get_config(self):
        config = {
            "optimizer": self.optimizer,
            "start_averaging": self.start_averaging,
            "average_period": self.average_period,
        }
        base_config = super(StochasticWeightAveraging, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @property
    def lr(self):
        return self.optimizer.lr





# """




