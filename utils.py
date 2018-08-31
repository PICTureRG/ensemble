import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.python.client import timeline
from tensorflow.python.ops import math_ops
from tensorflow.python.summary import summary
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.lib.io import file_io
from tensorflow.core.protobuf import config_pb2

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
from tensorflow.python.training import queue_runner

import time, os, sys
import numpy

FLAGS = tf.app.flags.FLAGS

# Returns readable date and computer date
def get_date():
    return os.popen("date").read().strip() + ", " + os.popen("date +%s").read().strip()

def get_config():
    return tf.ConfigProto(inter_op_parallelism_threads=32,
                          intra_op_parallelism_threads=32,
                          allow_soft_placement=True)

def psutil_supported():
    return FLAGS.cluster == "xsede" or FLAGS.cluster == "titan"


#NOTE: average_gradients function written by Tensorflow developers in
#Cifar10 multi-gpu example
def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    
    Note that this function provides a synchronization point across all towers.
    
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
    is over individual gradients. The inner list is over the gradient
    calculation for each tower.
    Returns:
    List of pairs of (gradient, variable) where the gradient has been averaged
    across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)
            
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)
            
        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)#Had to swap
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def get_preprocessed_queue(dataset, train_image_size):
    img_pre_fn = preprocessing_factory.get_preprocessing(FLAGS.preprocessing_name, 
                                                         is_training=True)
    with tf.device("/cpu:0"):
        with tf.name_scope("reading"):
            data_provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset, num_readers=FLAGS.num_data_readers,
                common_queue_capacity=20*FLAGS.batch_size,
                common_queue_min=10*FLAGS.batch_size)
            [image, label] = data_provider.get(['image', 'label'])
        with tf.name_scope("to-preprocessing"):
            capacity = 10 * FLAGS.batch_size
            to_pre_queue = data_flow_ops.FIFOQueue(capacity=capacity,
                                                   dtypes=[image.dtype, label.dtype],
                                                   shapes=None,
                                                   name="to_pre_queue")
            to_pre_op = to_pre_queue.enqueue([image, label])
            queue_runner.add_queue_runner(queue_runner.QueueRunner(to_pre_queue, [to_pre_op] * 10))
            tf.summary.scalar("to_pre_fraction_of_%d_full" % capacity,
                            math_ops.to_float(to_pre_queue.size()) * (1. / capacity))
            image, label = to_pre_queue.dequeue()
        with tf.name_scope("preprocessing"):
            image = img_pre_fn(image, train_image_size, train_image_size, fast_mode=FLAGS.fast_mode)
    # with tf.device('/gpu:0'):
        with tf.name_scope("done-preprocessing"):
            capacity = 10 * FLAGS.batch_size
            done_pre_queue = data_flow_ops.FIFOQueue(capacity=capacity,
                                                   dtypes=[image.dtype, label.dtype],
                                                   shapes=[[train_image_size, train_image_size, 3], []],
                                                   name="done_pre_queue")
            done_pre_op = done_pre_queue.enqueue([image, label])
            queue_runner.add_queue_runner(queue_runner.QueueRunner(done_pre_queue, [done_pre_op] * 10))
            tf.summary.scalar("done_pre_fraction_of_%d_full" % capacity,
                           math_ops.to_float(done_pre_queue.size()) * (1. / capacity))
    return done_pre_queue

