#Author: rbpittma
#Description: Simplified trainer script for the models provided by slim.
#Does not include learning rate decay

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()
# import os
# os.environ["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"] + ":/usr/local/cuda/extras/CUPTI/lib64"
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
import tensorflow.contrib.slim as slim
# from cpu_logger import CPULogger

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

import time, sys
import numpy

#Integers
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size")
tf.app.flags.DEFINE_integer("num_gpus", 1, "Number of gpus")
tf.app.flags.DEFINE_integer("max_steps", 1000, "Number of steps")
tf.app.flags.DEFINE_integer("log_every_n_steps", 1, "Logging frequency")
tf.app.flags.DEFINE_integer("save_summaries_steps", 100, "Summary saving frequency")
tf.app.flags.DEFINE_integer("num_data_readers", 20, "Number of data reader threads")
tf.app.flags.DEFINE_integer("num_batching_threads", 20, "Number of batching enqueue threads")
tf.app.flags.DEFINE_integer("batch_queue_size", 20, "Number of batches in the batching queue")
tf.app.flags.DEFINE_integer("trace_every_n_steps", 100, "Trace every n steps")

#Floats
tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
tf.app.flags.DEFINE_float("weight_decay", 0.00004, 'The weight decay on the model weights.')

#Strings
tf.app.flags.DEFINE_string("cluster", "xsede", "String specification of cluster code is running on (xsede or summit)")

train_dir = ""
tf.app.flags.DEFINE_string("train_dir", "~/", "Training directory")
tf.app.flags.DEFINE_string("dataset_dir",
                           "/lustre/atlas/proj-shared/csc160/rbpittma_tensor/imagenet-data-lin",
                           "Dataset directory")
tf.app.flags.DEFINE_string("mode", "train", "Script mode")
#train: run normal training
#preprocessing: Get profiling data on preprocessor throughput
#compute: Get profiling data on GPU compute throughput

#tf.app.flags.DEFINE_string("checkpoint_dir", "~/", "Directory where checkpoint will be saved or loaded from")
tf.app.flags.DEFINE_string("dataset_name", "imagenet", "Name of dataset")
tf.app.flags.DEFINE_string("preprocessing_name", "inception_v1", "Name of preprocessing")
tf.app.flags.DEFINE_string("model_name", 'inception_v1', 'The name of the architecture to train.')

#Bool
tf.app.flags.DEFINE_boolean("fast_mode", False, 'Whether to do faster preprocessing')
# tf.app.flags.DEFINE_boolean("resume", False, 'Whether to resume from a checkpoint file')

FLAGS = tf.app.flags.FLAGS

from utils import *

def get_tower_batches(dataset, train_image_size):
    done_pre_queue = get_preprocessed_queue(dataset, train_image_size)
    return [done_pre_queue.dequeue() for i in range(FLAGS.num_gpus)]

def get_precessing_eater(dataset, train_image_size):
    done_pre_queue = get_preprocessed_queue(dataset, train_image_size)
    image, label = done_pre_queue.dequeue()
    batch_out = tf.train.batch(
        [image, label],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_batching_threads,
        capacity=FLAGS.batch_queue_size * FLAGS.batch_size,
        shapes=[image.get_shape(), []])
    
    with tf.control_dependencies(batch_out):
        eater = tf.no_op()
    # eater = tf.Print(eater, [])
    return eater


# def save_float_summaries(data, writer):
#   for i in range(len(data)):
#     summary_i = tf.Summary(value=[tf.Summary.Value(tag="cpu_util", simple_value=data[i])])

# def import_psutil():
#     if FLAGS.cluster == "xsede" or FLAGS.cluster == "titan":
#         global psutil
#         import psutil
#         print("imported psutil")

# def cpu_util_summary():
#     if FLAGS.cluster == "xsede" or FLAGS.cluster == "titan":#psutil needs to be installed.
#         def get_util():
#             return numpy.float32(psutil.cpu_percent())
#         cpu_util_tensor = tf.py_func(get_util, [], tf.float32)
#         tf.summary.scalar("cpu_util", cpu_util_tensor)


def compute_loss(images, labels, num_classes, network_fn, scope=None, gpu_idx=0):
    with tf.device("/device:GPU:%d" % gpu_idx):
    
        # summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        logits, end_points = network_fn(images)#, reuse=gpu_idx!=0)
        
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar("train-accuracy", acc)
        
        tf.losses.softmax_cross_entropy(labels, logits)
        losses = tf.get_collection(tf.GraphKeys.LOSSES, scope)
        #Ignoring other types of losses...
        loss = tf.add_n(losses, name="loss")
        tf.summary.scalar("loss", loss)
    return loss



def get_main_ops():
    # import_psutil()
    os.system("sh gpu_util.sh > " + train_dir + "/nvidia_smi_reports.log 2>&1 &")

    # cpu_logger = CPULogger()
    # cpu_logger.start()
    if FLAGS.cluster == "summit":
        tf.load_file_system_library("/usr/local/cuda/extras/CUPTI/lib64/libcupti.so.9.0")
        # print(os.popen("ls /usr/local/cuda/extras/CUPTI/lib64").read())
        
    main_start_t = time.time()
    
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, "train", FLAGS.dataset_dir)
    
    network_fn = nets_factory.get_network_fn(FLAGS.model_name, 
                                             num_classes=dataset.num_classes,
                                             weight_decay=FLAGS.weight_decay, 
                                             is_training=True)
    train_image_size = network_fn.default_image_size
    print("Model:", FLAGS.model_name)
    #Default graph, default to using cpu
    #with tf.Graph().as_default(), tf.device('/cpu:0'):
    with tf.get_default_graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
        if FLAGS.num_gpus > 1:
            all_grads = []
            tower_batches = get_tower_batches(dataset, train_image_size)
            # done_pre_queue = get_preprocessed_queue(dataset, train_image_size)
            #Get gradients for each tower
            for gpu_idx in range(FLAGS.num_gpus):
                with tf.device("/device:GPU:%d" % gpu_idx):
                    with tf.name_scope("tower_%d" % gpu_idx) as scope:
                        # image, label = done_pre_queue.dequeue()
                        image, label = tower_batches[gpu_idx]
                        loss = compute_loss(image, label, dataset.num_classes, network_fn, scope, gpu_idx)
                        tf.get_variable_scope().reuse_variables()
                        #Summaries on first GPU
                        # if gpu_idx == 0:
                        #     summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                        my_grads = optimizer.compute_gradients(loss)#, colocate_gradients_with_ops=True)
                        all_grads.append(my_grads)
            avg_grads = average_gradients(all_grads)
            train_op = optimizer.apply_gradients(avg_grads)
        else:
            # Optimized single GPU code
            done_pre_queue = get_preprocessed_queue(dataset, train_image_size)
            image, label = done_pre_queue.dequeue()
            images, labels = tf.train.batch(
                [image, label],
                batch_size=FLAGS.batch_size,
                num_threads=FLAGS.num_batching_threads,
                capacity=FLAGS.batch_queue_size * FLAGS.batch_size,
                shapes=[image.get_shape(), []])
            labels = slim.one_hot_encoding(labels, dataset.num_classes)
            
            with tf.device("/device:GPU:0"):
                with tf.name_scope("compute") as scope:
                    loss = compute_loss(images, labels, dataset.num_classes, network_fn, scope, 0)
                    tf.get_variable_scope().reuse_variables()
                    my_grads = optimizer.compute_gradients(loss)#, colocate_gradients_with_ops=True)
                train_op = optimizer.apply_gradients(my_grads)
                

        #Extra summaries
        # summaries.append(tf.summary.scalar('learning_rate', lr))
        # tf.summary.scalar('learning_rate', lr)
        # cpu_util_summary()
        
        # def get_gpu_util(idx):
        #     util_level = int(os.popen("nvidia-smi -i %d --format=csv --query-gpu=utilization.gpu | tail -n 1 | egrep -o [0-9]+" % idx).read().strip())
        #     return numpy.float32(float(util_level))
        # for i in range(FLAGS.num_gpus):
        #     gpu_util_tensor = tf.py_func(get_gpu_util, [i], tf.float32)
        #     tf.summary.scalar("gpu %d util" % i, gpu_util_tensor)
            
        # def get_gpu_power(idx):
        #     power = int(os.popen("nvidia-smi -i %d --format=csv --query-gpu=power.draw | tail -n 1 | egrep -o [0-9]+ | head -n 1" % idx).read().strip())
        #     return numpy.float32(float(power))
        # for i in range(FLAGS.num_gpus):
        #     gpu_power_tensor = tf.py_func(get_gpu_power, [i], tf.float32)
        #     tf.summary.scalar("gpu %d power (Watts)" % i, gpu_power_tensor)

        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summary_op = tf.summary.merge(summaries)#CHECK OVER
        # summary_op = tf.group(summary_op, train_op)
        #Lastly, create initializer
        init = tf.global_variables_initializer()
    return init, train_op, summary_op

def train():
    os.system("mpstat -P ALL 1 > " + train_dir + "/cpu_util.log 2>&1 &")

    main_start_t = time.time()
    init, train_op, summary_op = get_main_ops()
    sess = tf.Session(config=get_config())
    sess.run(init)
    # coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess=sess)#, coord=coord)
    #Loss can be printed using tf.Print and some global step math
    summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
    print("%-8s%-20s%-20s%-20s" % ("Step", "Compute time", "Full step time", "Extra time"))
    for step in range(1, FLAGS.max_steps+1):
        trace_run_options = None
        run_metadata = None
        if step % FLAGS.trace_every_n_steps == 0:
            print("Grabbing trace")
            trace_run_options = config_pb2.RunOptions(trace_level=config_pb2.RunOptions.FULL_TRACE)
            run_metadata = config_pb2.RunMetadata()
        start_time = time.time()
        # done = False
        # while not done:
        #     try:
        if step % FLAGS.save_summaries_steps == 0:
            _, summary_str = sess.run([train_op, summary_op],
                                      options=trace_run_options,
                                      run_metadata=run_metadata)
            summary_writer.add_summary(summary_str, step)
            print("Added summary")
        else:
            sess.run(train_op,
                     options=trace_run_options,
                     run_metadata=run_metadata)
            # done = True
            # except tf.errors.OutOfRangeError:
            #     print("Encountered Out of range error")
        end_time = time.time()
        # if step % FLAGS.save_summaries_steps == 0:
        #     summary_str = sess.run(summary_op)
        #     summary_writer.add_summary(summary_str, step)
        #     print("Added summary")
        if step % FLAGS.trace_every_n_steps == 0:
            try: 
                tl = timeline.Timeline(run_metadata.step_stats)
                trace = tl.generate_chrome_trace_format()
                trace_filename = os.path.join(train_dir, 'tf_trace-step-%d.json' % step)
                print('Writing trace to %s' % trace_filename)
                file_io.write_string_to_file(trace_filename, trace)
                summary_writer.add_run_metadata(run_metadata, 'run_metadata-step-%d' % step)
            except Exception:
                print("ERROR occurred in timeline save")
        if step == 1:
            print("Startup time =", round(time.time() - main_start_t, 4))
            print("Startup finished at:", get_date())
        if step % FLAGS.log_every_n_steps == 0:
            # print("Step %8d: Compute time = %8.4f | Full step time = %8.4f" % 
            comp_time = round(end_time - start_time, 4)
            full_time = round(time.time()-start_time, 4)
            extra = round(full_time - comp_time, 4)
            print("%-8d%-20f%-20f%-20f" % (step, comp_time, full_time, extra))
            sys.stdout.flush()
        # cpu_logger.stop()
        # save_float_summaries(cpu_logger.get_utils(), summary_writer)
        # print("Saved cpu utilization summaries")
    # coord.request_stop()
    # coord.join(stop_grace_period_secs=0.05)
    main_end_t = time.time()
    print("Completed training in", round(main_end_t - main_start_t, 4), "seconds")
    print("Completed training at:", get_date())
    # sess.close()
    # print("CLOSED SESSION")
    sys.stdout.flush()
    time.sleep(60)

def measure_pre_throughput():
    # import_psutil()
    
    if FLAGS.cluster == "summit":
        tf.load_file_system_library("/usr/local/cuda/extras/CUPTI/lib64/libcupti.so.9.0")
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, "train", FLAGS.dataset_dir)
    network_fn = nets_factory.get_network_fn(FLAGS.model_name, 
                                             num_classes=dataset.num_classes,
                                             weight_decay=FLAGS.weight_decay, 
                                             is_training=True)
    train_image_size = network_fn.default_image_size
    print("Model:", FLAGS.model_name)
    #Default graph, default to using cpu
    with tf.get_default_graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.Variable(0, name="global_step", trainable=False)
        eater = get_precessing_eater(dataset, train_image_size)
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summary_op = tf.summary.merge(summaries)
        init = tf.global_variables_initializer()
        # cpu_util_summary()
        
    sess = tf.Session(config=get_config())
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(init)
    summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
    print("Warming up")
    sys.stdout.flush()
    #Warmup
    for i in range(10):
        sess.run(eater)
    print("Profiling")
    sys.stdout.flush()
    times = []
    num_batches = 10
    for i in range(num_batches):
        start = time.time()
        sess.run(eater)
        end = time.time()
        times.append(end-start)
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, i+1)
    avg_time = sum(times) / len(times)
    batches_per_sec = 1./avg_time
    images_per_sec = batches_per_sec * FLAGS.batch_size
    print("Batches/sec =", batches_per_sec)
    print("Images/sec =", images_per_sec)
    sys.stdout.flush()
    # coord.request_stop()

def measure_compute_throughput():
    init, train_op, summary_op = get_main_ops()
    sess = tf.Session(config=get_config())
    sess.run(init)
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess=sess, coord=coord)
    #Loss can be printed using tf.Print and some global step math
    summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
    num_warmup = 100
    print("Warming up %d steps" % num_warmup)
    sys.stdout.flush()
    for warmup in range(num_warmup):
        sess.run(train_op)
    times = []
    
    print("Benchmarking compute")
    fill_sleep = 60
    print("Letting queues fill for %d seconds" % fill_sleep)
    sys.stdout.flush()
    time.sleep(fill_sleep)
    
    num_test_batches = 100 # should be less than FLAGS.batch_queue_size
    print("Benchmarking %d batches" % num_test_batches)
    for step in range(num_test_batches):
        # trace_run_options = None
        # run_metadata = None
        # if step % FLAGS.trace_every_n_steps == 0:
        #     print("Grabbing trace")
        #     trace_run_options = config_pb2.RunOptions(trace_level=config_pb2.RunOptions.FULL_TRACE)
        #     run_metadata = config_pb2.RunMetadata()
            
        start_time = time.time()
        sess.run(train_op)
        # sess.run(train_op,
        #          options=trace_run_options,
        #          run_metadata=run_metadata)
        
        end_time = time.time()
        times.append(end_time - start_time)
        #Extra time required to run summary op doesn't matter
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step+1)
        
        # if step % FLAGS.trace_every_n_steps == 0:
        #     try: 
        #         tl = timeline.Timeline(run_metadata.step_stats)
        #         trace = tl.generate_chrome_trace_format()
        #         trace_filename = os.path.join(train_dir, 'tf_trace-step-%d.json' % step)
        #         print('Writing trace to %s' % trace_filename)
        #         file_io.write_string_to_file(trace_filename, trace)
        #         summary_writer.add_run_metadata(run_metadata, 'run_metadata-step-%d' % step)
        #     except Exception:
        #         print("ERROR occurred in timeline save")
        time.sleep(1)

    avg_time = sum(times) / len(times)
    batches_per_sec = 1./avg_time
    images_per_sec = batches_per_sec * FLAGS.batch_size
    # print("Batches/sec =", batches_per_sec)
    print("Images/sec =", images_per_sec)
    print("Produced by times:", times)
    summary_writer.flush()
    time.sleep(10)
    # coord.request_stop()

def print_flags():
    print("Batch size:", FLAGS.batch_size)
    #If titan, multi node training may be in place, so create own log dir
    if FLAGS.cluster == "titan":
        global train_dir
        train_dir = os.path.join(FLAGS.train_dir, "rank_" + str(rank))
        os.system("mkdir " + train_dir)
        print("Rank %d hostname: %s" % (rank, os.popen("hostname").read().strip()))
    else:#else just use the flags dir.
        train_dir = FLAGS.train_dir
    print("Log directory:", train_dir)
    print("Number of GPUs:", FLAGS.num_gpus)
    print("Model:", FLAGS.model_name)
    print("Cluster:", FLAGS.cluster)
    print("Mode:", FLAGS.mode)
    print("Fast preprocessing mode:", FLAGS.fast_mode)
    print("Hostname:", os.popen("hostname").read())
    if FLAGS.cluster == "xsede":
        print("nvidia-smi output")
        print("=================")
        print(os.popen("nvidia-smi").read())
        sys.stdout.flush()

def main(argv):
    print("Main started at:", get_date())
    if FLAGS.cluster == "titan" and FLAGS.num_gpus > 1:
        print("Titan cluster does not support multi-gpu per node training")
    else:
        print_flags()
        if FLAGS.mode == "train":
            train()
        elif FLAGS.mode == "preprocessing":
            measure_pre_throughput()
        elif FLAGS.mode == "compute":
            measure_compute_throughput()
        else:
            print("Error: Unrecognized FLAGS.mode option:", FLAGS.mode)

if __name__ == "__main__":
    tf.app.run()
    
