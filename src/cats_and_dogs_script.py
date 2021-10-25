import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing.image import DirectoryIterator
from tensorflow.keras.models import Sequential
from datetime import datetime
from cnn import Cnn

dt = datetime.now()
print('start_at:', dt.strftime('%Y-%m-%d %H:%M:%S'))

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(ROOT_PATH, '../cats_and_dogs')
CHART_DIR = os.path.join(ROOT_PATH, '../chart')
RS_DIR = os.path.join(ROOT_PATH, '../evaluate')

train_dir = os.path.join(DATA_DIR, 'training_set')
valid_dir = os.path.join(DATA_DIR, 'validation_set')
test_dir = os.path.join(DATA_DIR, 'test_set')


def run(seq: int, dropout: float, learning_rate: float, epoch: int):
    cnn = Cnn(model=Sequential(), dropout=dropout)
    cnn.compile_model(learning_rate=learning_rate)
    history = cnn.train(train_data=train_set, valid_data=valid_set, epochs=epoch)
    now = datetime.now()
    seq_dir = os.path.join(CHART_DIR, str(seq))
    if not os.path.exists(seq_dir):
        os.mkdir(seq_dir)

    filename = 'cnn_dr={dr}_lr={lr}_ep={ep}'.format(dr=str(dropout), lr=str(learning_rate), ep=str(epoch))
    cnn.save_plot(history, os.path.join(seq_dir, filename))
    evaluate = cnn.evaluate_model(test_data=test_set)

    cnn.save_evaluate(seq, os.path.join(ROOT_PATH, '../evaluate'), {
        'dropout': dropout,
        'learning_rate': learning_rate,
        'epoch': epoch,
        'evaluate': {
            'loss': evaluate[0],
            'acc': evaluate[1]
        }
    })

    print('loss: {}, acc: {}'.format(*evaluate))

    predict = cnn.predict_model(test_data=test_set)
    cnn.save_predict(seq, os.path.join(ROOT_PATH, '../'), predict)
    del cnn

    return {
        'evaluate': evaluate,
        'predict': predict
    }


if __name__ == '__main__':
    # image generate for Data Argumentation
    img_generator = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Generate train data
    train_set: DirectoryIterator = img_generator.flow_from_directory(
        train_dir,
        target_size=(128, 128),
        batch_size=30
    )

    # Generate test data
    # raw image
    test_generator = ImageDataGenerator(
        rescale=1. / 255
    )

    # Generate validation data
    valid_set: DirectoryIterator = test_generator.flow_from_directory(
        valid_dir,
        target_size=(128, 128),
        batch_size=30
    )

    test_set: DirectoryIterator = test_generator.flow_from_directory(
        test_dir,
        target_size=(128, 128),
        batch_size=30,
        shuffle=False
    )

    # dropout_list = [0.3]
    # lr_list = [0.0001]
    # epoch_list = [2]

    # 데이터 수집을 위한 케이스 세팅
    # dropout: 5
    # learning_rate: 6
    # epoch: 1(50)
    # total: 30 * 50 = 150

    # 2021.10.24
    # 현재 가장 높은 정확도
    # learning_rate 0.0007
    # dropout 0.3
    # epoch

    dropout_list = [0.3]
    lr_list = [0.0007]
    epoch_list = [40]

    # final...
    SEQ: int = 0

    for dr in dropout_list:
        for lr in lr_list:
            for ep in epoch_list:
                SEQ += 1
                print('----------' * 5)
                print('SEQ', SEQ)
                print('dropout:', dr)
                print('learning_rate:', lr)
                print('epoch:', ep)
                rs = run(seq=SEQ, dropout=dr, learning_rate=lr, epoch=ep)
                ev = rs['evaluate']
                pr = rs['predict']
                print('----------' * 5)
    end_at = datetime.now()
    print('end_at:', end_at.strftime('%Y-%m-%d %H:%M:%S'))
    duration = end_at - dt
    print('duration:', duration.seconds)
