import os
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.python.keras.preprocessing.image import DirectoryIterator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import History
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

# OMP: Error #15: Initializing libiomp5md.dll
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Cnn:
    _ROOT_PATH: str
    _model: Sequential

    def __init__(self, model: Sequential, dropout: float = 0.3):
        self._model = self.config_model(model, dropout)

    @staticmethod
    def config_model(model, dropout: float = 0.3):
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
        # tf.keras.layers.Dropout(0.3),
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.BatchNormalization())

        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        # tf.keras.layers.Dropout(0.3),
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.BatchNormalization())

        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        # model.add(layers.Dropout(0.3))
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.BatchNormalization())

        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        # model.add(layers.Dropout(0.3))
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.BatchNormalization())

        # Flatten the results to feed into a DNN
        model.add(layers.Flatten())
        # 512 neuron hidden layer
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(dropout))
        model.add(layers.Dense(2, activation='sigmoid'))

        model.summary()

        return model

    def compile_model(self, learning_rate: float = 0.1):
        self._model.compile(
            optimizer=RMSprop(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['acc']
        )

    def train(self, train_data: DirectoryIterator, valid_data: DirectoryIterator, epochs: int = 1):
        print('----------' * 5)
        print('start train')
        history = self._model.fit(
            train_data,
            epochs=epochs,
            verbose=1,
            validation_data=valid_data
        )
        print('----------' * 5)

        return history

    @staticmethod
    def save_plot(history: History, filename: str):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        print('history len: ', len(acc))
        epochs = range(len(acc))

        plt.plot(epochs, acc, label='train')
        plt.plot(epochs, val_acc, label='valid')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='upper left')
        plt.savefig(filename + '_acc.png')
        print('save chart...', filename + '_acc.png')
        plt.clf()

        plt.plot(epochs, loss, label='train')
        plt.plot(epochs, val_loss, label='valid')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='upper left')
        plt.savefig(filename + '_loss.png')
        print('save chart...', filename + '_loss.png')
        plt.clf()

    def evaluate_model(self, test_data: DirectoryIterator):
        print('----------' * 5)
        print('start evaluate')
        print('----------' * 5)
        return self._model.evaluate(test_data)

    @staticmethod
    def save_evaluate(seq: int, ev_dir: str, evaluate: dict):
        with open(os.path.join(ev_dir, 'evaluate_{SEQ}.json'.format(SEQ=seq)), 'w') as f:
            json.dump(evaluate, f)

    def predict_model(self, test_data: DirectoryIterator):
        print('----------' * 5)
        print('start predict')
        print('----------' * 5)
        label = ['cat', 'dog']

        predict = self._model.predict(test_data)
        result = np.argmax(predict, axis=-1)

        match_predict = {'filename': [], 'predict': []}
        for i, filename in enumerate(test_data.filenames):
            match_predict['filename'].append(filename)
            match_predict['predict'].append(label[result[i]])

        predict_df = pd.DataFrame.from_dict(match_predict)

        return predict_df

    @staticmethod
    def save_predict(seq: int, pr_dir: str, predict_df: pd.DataFrame):
        path = os.path.join(pr_dir, str(seq))
        # pycharm 경고 메시지 보기 싫어서...
        csv = predict_df.to_csv()
        with open('{0}.csv'.format(path), 'w') as f:
            f.write(csv)
