import numpy as np
import tensorflow as tf
from tensorflow.contrib.memory_stats.python.ops.memory_stats_ops import BytesLimit, BytesInUse, MaxBytesInUse
from time import time
import math

from include.data import get_data_set, resize_data
from include.model import s_model, lr


#Params
global_accuracy = 0
global_avg_bs = 0
global_counter = 0
global_model = "small"

_batch_size = 10 #Max Batch_Size=14895
_epoch = 10


#parse arguments
import optparse
parser = optparse.OptionParser()
parser.add_option('-m', '--model', type=str , dest="model", default="small", help="define whatever model should be run (small or big)")
parser.add_option('-i', '--input_size', type=int , dest="input_size", default=32, help="define the Input size as integer. Default value is 32")
parser.add_option('-b', '--batch_size', type=int , dest="batch_size", default=128, help="define the batch size. Default value is 128")
parser.add_option('-e', '--epoch', type=int , dest="epoch", default=10, help="define the Epoch number. Default value is 10")
parser.add_option('-g', '--gpu_array', type=str, dest="gpu_array", default='0,1,1', help="define on which GPUs should the model be splited. Default value is '0,1,1'")

options, args = parser.parse_args()
global_model = options.model
_batch_size = options.batch_size
_epoch = options.epoch
_input_size = options.input_size

train_x, train_y = get_data_set("train")
test_x, test_y = get_data_set("test")

#Resize the dataset
_train_size = train_x.shape[0]
_test_size = test_x.shape[0]
train_x = resize_data(train_x, _input_size, _input_size, _train_size)
train_y = train_y[:_train_size, :]

test_x = resize_data(test_x, _input_size, _input_size, _test_size)
test_y = test_y[:_test_size, :]

#Set Model to use
if(global_model == "small"):
    x, y, output, y_pred_cls, global_step, learning_rate = s_model()

#PARAMS
_SAVE_PATH = "./tensorboard/cifar-10-v1.0.0/"


# LOSS AND OPTIMIZER
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                   beta1=0.9,
                                   beta2=0.999,
                                   epsilon=1e-08).minimize(loss, global_step=global_step)


# PREDICTION AND ACCURACY CALCULATION
correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# SAVER
merged = tf.summary.merge_all()
saver = tf.train.Saver()
session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
session_config.gpu_options.allow_growth = True
sess = tf.Session(config=session_config)
train_writer = tf.summary.FileWriter(_SAVE_PATH, sess.graph)


'''try:
    print("\nTrying to restore last checkpoint ...")
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=_SAVE_PATH)
    saver.restore(sess, save_path=last_chk_path)
    print("Restored checkpoint from:", last_chk_path)
except ValueError:
    print("\nFailed to restore checkpoint. Initializing variables instead.")
    sess.run(tf.global_variables_initializer())
'''
sess.run(tf.global_variables_initializer())


def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])


def train(epoch):
    #run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    batch_size = int(math.ceil(_train_size / _batch_size))
    i_global = 0
    global global_avg_bs 
    global global_counter

    for s in range(batch_size):
        batch_xs = train_x[s*_batch_size: (s+1)*_batch_size]
        batch_ys = train_y[s*_batch_size: (s+1)*_batch_size]

        start_time = time()
        i_global, _, batch_loss, batch_acc = sess.run(
            [global_step, optimizer, loss, accuracy],
            feed_dict={x: batch_xs, y: batch_ys, learning_rate: lr(epoch)})
        duration = time() - start_time

        if s % 100 == 0:
            percentage = int(round((s/batch_size)*100))
            #print(percentage)
            bar_len = 29
            filled_len = int((bar_len*int(percentage))/100)
            bar = '=' * filled_len + '>' + '-' * (bar_len - filled_len)

            msg = "Global step: {:>5} - [{}] {:>3}% - acc: {:.4f} - loss: {:.4f} - {:.1f} sample/sec"
            print(msg.format(i_global, bar, percentage, batch_acc, batch_loss, _batch_size / duration))
            global_avg_bs += (_batch_size / duration)
            global_counter += 1 

    #Print GPUs usage
    '''with tf.device('/device:GPU:0'):
        bytes_in_use = MaxBytesInUse()
        print("memory usage on GPU0: ", convert_size(sess.run(bytes_in_use)))
    with tf.device('/device:GPU:1'):
        bytes_in_use = MaxBytesInUse()
        print("memory usage on GPU1: ", convert_size(sess.run(bytes_in_use)))
    with tf.device('/device:GPU:2'):
        bytes_in_use = MaxBytesInUse()
        print("memory usage on GPU2: ", convert_size(sess.run(bytes_in_use)))
    with tf.device('/device:GPU:3'):
        bytes_in_use = MaxBytesInUse()
        print("memory usage on GPU3: ", convert_size(sess.run(bytes_in_use)))
    '''

    test_and_save(i_global, epoch)


def test_and_save(_global_step, epoch):
    global global_accuracy

    i = 0
    predicted_class = np.zeros(shape=_test_size, dtype=np.int)
    while i < _test_size:
        j = min(i + _batch_size, _test_size)
        batch_xs = test_x[i:j, :]
        batch_ys = test_y[i:j, :]
        predicted_class[i:j] = sess.run(
            y_pred_cls,
            feed_dict={x: batch_xs, y: batch_ys, learning_rate: lr(epoch)}
        )
        i = j

    correct = (np.argmax(test_y, axis=1) == predicted_class)
    acc = correct.mean()*100
    correct_numbers = correct.sum()

    mes = "\nEpoch {} - accuracy: {:.2f}% ({}/{})"
    print(mes.format((epoch+1), acc, correct_numbers, _test_size))

    if global_accuracy != 0 and global_accuracy < acc:

        summary = tf.Summary(value=[
            tf.Summary.Value(tag="Accuracy/test", simple_value=acc),
        ])
        train_writer.add_summary(summary, _global_step)

        saver.save(sess, save_path=_SAVE_PATH, global_step=_global_step)

        mes = "This epoch receive better accuracy: {:.2f} > {:.2f}. Saving session..."
        print(mes.format(acc, global_accuracy))
        global_accuracy = acc

    elif global_accuracy == 0:
        global_accuracy = acc

    print("###########################################################################################################")


def main():
    for i in range(_epoch):
        print("\nEpoch: {0}/{1}\n".format((i+1), _epoch))
        train(i)
    print("AVG sample/sec: ", global_avg_bs/global_counter)

if __name__ == "__main__":
    main()


sess.close()
