#%load_ext tensorboard
#%reload_ext tensorboard

from tensorboard.plugins.hparams import api as hp
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([32, 256]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.8))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd', 'rmsprop']))

METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )

def train_test_model(hparams):
    model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(max_words, 256, input_length=256),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(hparams[HP_NUM_UNITS], kernel_regularizer=tf.keras.regularizers.l2(0.001), activation=tf.nn.relu),
    tf.keras.layers.Dense(hparams[HP_NUM_UNITS], kernel_regularizer=tf.keras.regularizers.l2(0.001), activation=tf.nn.relu),
    tf.keras.layers.Dense(hparams[HP_NUM_UNITS], kernel_regularizer=tf.keras.regularizers.l2(0.001), activation=tf.nn.relu),
    tf.keras.layers.Dense(hparams[HP_NUM_UNITS], kernel_regularizer=tf.keras.regularizers.l2(0.001), activation=tf.nn.relu),
    tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid),
    ])
    model.compile(
      optimizer=hparams[HP_OPTIMIZER],
      loss='binary_crossentropy',
      metrics=['accuracy'],
    )

    model.fit(x_train, y_train, epochs=2) 
    _, accuracy = model.evaluate(x_test, y_test)
    return accuracy


def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        accuracy = train_test_model(hparams)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)


session_num = 0


for num_units in HP_NUM_UNITS.domain.values:
    for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
        for optimizer in HP_OPTIMIZER.domain.values:
            hparams = {
                HP_NUM_UNITS: num_units,
                HP_DROPOUT: dropout_rate,
                HP_OPTIMIZER: optimizer,
            }
            run_name = "run-%d" % session_num
            print('--- Starting trial: %s' % run_name)
            print({h.name: hparams[h] for h in hparams})
            run('logs/hparam_tuning/' + run_name, hparams)
            session_num += 1

#%tensorboard --logdir logs/hparam_tuning
#wget -q 'https://storage.googleapis.com/download.tensorflow.org/tensorboard/hparams_demo_logs.zip'
#unzip -q hparams_demo_logs.zip -d logs/hparam_demo
#%tensorboard --logdir logs/hparam_demo
