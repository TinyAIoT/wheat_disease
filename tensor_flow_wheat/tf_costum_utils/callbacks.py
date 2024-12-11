import time
import tensorflow as tf
import logging
import random

class BatchLoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self, batch_size, train_sample_count, epochs, interval_s = 10, ensure_determinism=False):
        # train_sample_count could be smaller than the batch size, so make sure total_batches is atleast
        # 1 to avoid a 'divide by zero' exception in the 'on_train_batch_end' callback.
        self.total_batches = max(1, int(train_sample_count / batch_size))
        self.last_log_time = time.time()
        self.epochs = epochs
        self.interval_s = interval_s
        print(f'Using batch size: {batch_size}', flush=True)

    # Within each epoch, print the time every 10 seconds
    def on_train_batch_end(self, batch, logs=None):
        current_time = time.time()
        if self.last_log_time + self.interval_s < current_time:
            print('Epoch {0}% done'.format(int(100 / self.total_batches * batch)), flush=True)
            self.last_log_time = current_time

    # Reset the time the start of every epoch
    def on_epoch_end(self, epoch, logs=None):
        self.last_log_time = time.time()

def scheduler(epoch, lr):
  if epoch < 5:
    return lr
  else:
    random_number = random.uniform(0.001, 0.1)
    lr = lr * tf.math.exp(random_number)
    return lr 
def get_lr_scheduler():
    callback=tf.keras.callbacks.LearningRateScheduler(scheduler)
    return callback
       
class LearningRatePrinter(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Print the learning rate after each epoch
        lr = round(self.model.optimizer.lr.numpy(), 5)
        logging.info(f"\n Epoch {epoch + 1}: Learning rate = {lr} \n")


def get_callback_list(callbacks,model):
    tf.keras.callbacks.CallbackList(
    callbacks=callbacks, add_history=False, add_progbar=False, model=model
)