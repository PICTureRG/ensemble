# This script constructs an duplicate DNN ensemble training session
# with an optimized preprocessing pipeline using Horovod. 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import horovod.tensorflow as hvd
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
from pipeline import Pipeline

from tensorflow.python.training import queue_runner

import time, os, sys, math
import numpy

from utils import *

#Integers
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size")
tf.app.flags.DEFINE_integer("max_steps", 1000, "Number of steps")
tf.app.flags.DEFINE_integer("log_every_n_steps", 1, "Logging frequency")
tf.app.flags.DEFINE_integer("save_summaries_steps", 1, "Summary saving frequency")
tf.app.flags.DEFINE_integer("num_data_readers", 10, "Number of data reader threads")
tf.app.flags.DEFINE_integer("num_batching_threads", 10, "Number of batching enqueue threads")
tf.app.flags.DEFINE_integer("batch_queue_size", 20, "Number of batches in the batching queue")
tf.app.flags.DEFINE_integer("trace_every_n_steps", 100, "Trace every n steps")
tf.app.flags.DEFINE_integer("num_pre", 1, "Number of preprocessors")
tf.app.flags.DEFINE_integer("num_workers", 2, "Number of workers")
tf.app.flags.DEFINE_integer("sleep_to_stop", 60, "Seconds to sleep before stopping")

#Floats
tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
tf.app.flags.DEFINE_float("weight_decay", 0.00004, 'The weight decay on the model weights.')

#Strings
tf.app.flags.DEFINE_string("mode", "train", "Specifies mode: ['train', 'preprocess']")
#train: run normal training
#preprocess: Get profiling data on preprocessor throughput

tf.app.flags.DEFINE_string("arch", "all_shared", "Architecture to use, one of [all_shared, single_bcast, multi_bcast]")
tf.app.flags.DEFINE_string("cluster", "xsede", "String specification of cluster code is running on (xsede or summit)")

tf.app.flags.DEFINE_string("train_dir", "~/", "Training directory")
tf.app.flags.DEFINE_string("dataset_dir",
                           "/lustre/atlas/proj-shared/csc160/rbpittma_tensor/imagenet-data-lin",
                           "Dataset directory")

#tf.app.flags.DEFINE_string("checkpoint_dir", "~/", "Directory where checkpoint will be saved or loaded from")
tf.app.flags.DEFINE_string("dataset_name", "imagenet", "Name of dataset")
tf.app.flags.DEFINE_string("preprocessing_name", "inception_v1", "Name of preprocessing")
tf.app.flags.DEFINE_string("model_name", 'inception_v1', 'The name of the architecture to train.')

#Bool
tf.app.flags.DEFINE_boolean("fast_mode", False, 'Whether to do faster preprocessing')

FLAGS = tf.app.flags.FLAGS

def train():
    main_start_t = time.time()
    pipeline = Pipeline(FLAGS)
    pipeline.start_mpstat()
    with tf.get_default_graph().as_default(), tf.device('/gpu:0'):
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
        loss = pipeline.compute_loss()
        grads = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads)
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summary_op = tf.summary.merge(summaries)
        init = tf.global_variables_initializer()
    #Generic TF running section of code:
    sess = tf.Session(config=get_config())
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)#coord=coord)
    train_dir = pipeline.get_train_dir()
    summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
    startup_duration = 0
    if pipeline.get_rank() == 0:
        print("%-8s%-20s%-20s%-20s" % ("Step", "Compute time", "Full step time", "Extra time"))
    for step in range(1, FLAGS.max_steps+1):
        # os.system("free >> " + os.path.join(train_dir, "ensemble.txt"))
        trace_run_options = None
        run_metadata = None
        if step % FLAGS.trace_every_n_steps == 0:
            # print("Grabbing trace")
            trace_run_options = config_pb2.RunOptions(trace_level=config_pb2.RunOptions.FULL_TRACE)
            run_metadata = config_pb2.RunMetadata()
        start_time = time.time()
        if step % FLAGS.save_summaries_steps == 0:
            _, summary_str = sess.run([train_op, summary_op],
                                      options=trace_run_options,
                                      run_metadata=run_metadata)
            summary_writer.add_summary(summary_str, step)
            # print("Added summary")
        else:
            sess.run(train_op,
                     options=trace_run_options,
                     run_metadata=run_metadata)
        end_time = time.time()
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
            startup_duration = round(time.time() - main_start_t, 4)
            print("Startup time =", startup_duration)
            print("Startup finished at:", get_date())
        if step % FLAGS.log_every_n_steps == 0:
            comp_time = round(end_time - start_time, 4)
            full_time = round(time.time()-start_time, 4)
            extra = round(full_time - comp_time, 4)
            print("%-8d%-20f%-20f%-20f" % (step, comp_time, full_time, extra))
    main_end_t = time.time()
    training_time = round(main_end_t - main_start_t, 4)
    print("Completed training in", training_time, "seconds")
    print("Completed training at:", get_date())
    print("Without startup time:", round(training_time - startup_duration, 4))
    sys.stdout.flush()
    # coord.request_stop()
    # coord.join(threads)
    print("Sleeping", FLAGS.sleep_to_stop, "seconds")
    time.sleep(FLAGS.sleep_to_stop)
    
def measure_pre_throughput():
    main_start_t = time.time()
    pipeline = Pipeline(FLAGS)
    
    with tf.get_default_graph().as_default(), tf.device('/gpu:0'):
        global_step = tf.Variable(0, name="global_step", trainable=False)
        eater = pipeline.get_preprocessing_eater()
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summary_op = tf.summary.merge(summaries)
        init = tf.global_variables_initializer()
    #Generic TF running section of code:
    sess = tf.Session(config=get_config())
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)#coord=coord)
    # print(dir(threads[0]))
    summary_writer = tf.summary.FileWriter(pipeline.get_train_dir(), sess.graph)
    rank = pipeline.get_rank()
    if rank == 0: print("Warming up")
    sys.stdout.flush()
    #Warmup
    for i in range(100):
        sess.run(eater)
    print("Profiling")
    sys.stdout.flush()
    times = []
    num_batches = 100
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
    # print("Batches/sec =", batches_per_sec)
    print("Node: %d, images/sec =" % rank, images_per_sec)
    sys.stdout.flush()
    time.sleep(30)

def main(argv):
    print("Main started at:", get_date())
    if FLAGS.mode == "train":
        train()
    elif FLAGS.mode == "preprocess":
        measure_pre_throughput()
    else:
        print("Error: Unrecognized FLAGS.mode option:", FLAGS.mode)

if __name__ == "__main__":
    tf.app.run()
