import os
import stored

from keras_trainer import Trainer
from model_converters import KerasToTensorflow


# Configure the trainer.
trainer = Trainer(
    train_dataset_dir='/data/train',
    val_dataset_dir='/data/valid',
    output_model_dir='/output/model/keras',
    output_logs_dir='/output/logs',
    num_classes=2,
    epochs=1,
    batch_size=16,
    model_spec='mobilenet_v1',
)


# Run the trainer.
trainer.run()


# Convert the Keras HDF5 model to a Tensorflow model.
KerasToTensorflow.convert('/output/model/keras/best_model_max_acc.hdf5', '/output/model/tensorflow')


# ZIP up the Tensorflow model for serving.
stored.sync('/output/model/tensorflow', '/output/model/tensorflow.zip')
