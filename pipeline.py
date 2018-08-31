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

from tensorflow.python.training import queue_runner

import time, os, sys, math
import numpy

from utils import *

#Obtained manually, unfortunately, non-preprocessor nodes have no way
#to access the data type
post_pre_image_dtype = tf.float32
post_pre_label_dtype = tf.int64


### Single Broadcast explanation

# Things get tricky here...
# Observe that all ranks must have the same variable name for the
# data being passed around. This is because the allgathered data
# tensor must have the same name, but then since this data is fed
# into broadcast, the holder variables for rank 3 and 4 must use
# this name. 
# Update: Actually they don't have to take the same variable name, but
# instead we can manually specify the name in the hvd.operation(...,name=HERE)
# parameter.  
# Here's a pipeline visualization to help: (3 preprocessors, 5 workers, each preprocessor has data di, where i is its rank)

#         Dataflow illustration                         Tensor graph eval
# ========================================     ====================================
# 0: [d0]-,  ,-[d0,d1,d2]                      <<< allgather
#          \/
# 1: [d1]--||--[d0,d1,d2]                      <<< allgather
#          /\
# 2: [d2]-'  '-[d0,d1,d2]-,                    <<< broadcast on allgathered data (after queue dequeue) < 
#                         |\                                                                           \
# 3:                      | `-[d0,d1,d2]       <<< broadcast on holder variable with correct shape      < These 3 vars must have same name!
#                         |                                                                            /
# 4:                      `---[d0,d1,d2]       <<< broadcast on holder variable with correct shape     <


def create_qr(name, capacity, tensor_list, shapes, dtypes, num_threads, enqueue_many, dequeue_many, dequeue_amount=None):
    """
    name: string name to use for queue runner creation
    capacity: non-negative integer capacity of qr
    tensor_list: enqueue data
    shapes: shape of queue
    dtypes: 
    num_threads: number of running threads to use
    dequeue_many: bool - whether to dequeue many when returning new value
    enqueue_many: bool - whether to enqueue many when pushing to queue
    """
    with tf.name_scope(name):
        queue = data_flow_ops.FIFOQueue(capacity=capacity,
                                        dtypes=dtypes,
                                        shapes=shapes,
                                        name=name + "_queue")
        if enqueue_many:
            op = queue.enqueue_many(tensor_list)
        else:
            op = queue.enqueue(tensor_list)
        queue_runner.add_queue_runner(queue_runner.QueueRunner(queue, [op] * num_threads))
        tf.summary.scalar(name + "_fraction_of_%d_full" % capacity,
                          math_ops.to_float(queue.size()) * (1. / capacity))
        if dequeue_many:
            if dequeue_amount:
                data = queue.dequeue_many(dequeue_amount)
            else:
                raise ValueError("Need dequeue_amount specified if dequeue_many is true")
        else:
            data = queue.dequeue()
    return data

class Pipeline:
    ALL_SHARED = "all_shared"
    SINGLE_BCAST = "single_bcast"
    MULTI_BCAST = "multi_bcast"
    MPSTAT_DELAY_SECS = 1
    QR_THREADS = 10
    
    #PRE: arch is string
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self._set_arch()
        self.setup()
        self.print_flags()
        
    def _set_arch(self):
        self.is_all_shared   = False
        self.is_single_bcast = False
        self.is_multi_bcast  = False
        
        if self.FLAGS.arch == Pipeline.ALL_SHARED:
            self.is_all_shared   = True
        elif self.FLAGS.arch == Pipeline.SINGLE_BCAST:
            self.is_single_bcast = True
        elif self.FLAGS.arch == Pipeline.MULTI_BCAST:
            self.is_multi_bcast  = True
        else:
            raise ValueError("Invalid architecture \"%s\"" % self.FLAGS.arch)

    def print_flags(self):
        if self.rank == 0:
            print("Batch size:        ", self.FLAGS.batch_size)
            print("Log directory:     ", self.FLAGS.train_dir)
            print("Model:             ", self.FLAGS.model_name)
            print("Cluster:           ", self.FLAGS.cluster)
            print("Fast preprocessing:", self.FLAGS.fast_mode)
            print("num_hvd_send:      ", self.num_hvd_send)
            if self.is_single_bcast or self.is_multi_bcast:
                print("images_per_bcast:  ", self.images_per_bcast)


    def setup(self):
        # Init hvd differently depending on pipeline architecture
        if self.is_all_shared:
            hvd.init() # Should have imported different version of hvd
            self.rank = hvd.rank()
        elif self.is_single_bcast:
            group0 = list(range(self.FLAGS.num_pre))
            group1 = list(range(self.FLAGS.num_pre-1, self.FLAGS.num_workers))
            hvd.init([group0, group1])
            # print(group0, group1)
            # time.sleep(10)
            self.rank = hvd.global_rank()
            self.member_of_group = []#Which group(s) this rank belongs to
            self.group_rank_list = []#rank for each group
            
            if self.rank in group0:
                self.member_of_group.append(0)
                self.group_rank_list.append(hvd.rank(group=0))
            if self.rank in group1:
                self.member_of_group.append(1)
                self.group_rank_list.append(hvd.rank(group=1))
            # ASSERT: Only 1 rank should have 2 group ranks, the merger rank
        elif self.is_multi_bcast:
            groups = [list(range(self.FLAGS.num_workers)) for i in range(self.FLAGS.num_pre)]
            hvd.init(groups)
            self.rank = hvd.global_rank()
        # Take care of some other setup things
        
        # Training directory based on rank
        self.train_dir = os.path.join(self.FLAGS.train_dir, "rank_" + str(self.rank))
        os.system("mkdir " + self.train_dir)
        
        # Print hostname
        print("Rank %d hostname: %s" % (self.rank, os.popen("hostname").read().strip()))
        
        # amount of data to send in different stages
        self.num_hvd_send = 10 * self.FLAGS.batch_size // self.FLAGS.num_pre#hvd.global_size()
        if self.is_single_bcast:
            self.images_per_bcast = 5 * self.FLAGS.batch_size
        elif self.is_multi_bcast:
            self.images_per_bcast = self.FLAGS.batch_size
        
        self.dataset = dataset_factory.get_dataset(
            self.FLAGS.dataset_name, "train", self.FLAGS.dataset_dir)
    
        self.network_fn = nets_factory.get_network_fn(self.FLAGS.model_name, 
                                                      num_classes=self.dataset.num_classes,
                                                      weight_decay=self.FLAGS.weight_decay, 
                                                      is_training=True)
            
        self.train_image_size = self.network_fn.default_image_size
    
    def start_mpstat(self):
        # mpstat CPU logger
        os.system("mpstat -P ALL " + str(Pipeline.MPSTAT_DELAY_SECS) + " > " + self.train_dir + "/cpu_util.log 2>&1 &")
    
    def get_image_labels(self):
        if self.is_all_shared:
            ### ALL SHARED ###
            img_pre_fn = preprocessing_factory.get_preprocessing(self.FLAGS.preprocessing_name, 
                                                                 is_training=True)
            with tf.device("/cpu:0"):
                with tf.name_scope("reading"):
                    data_provider = slim.dataset_data_provider.DatasetDataProvider(
                        self.dataset, num_readers=self.FLAGS.num_data_readers,
                        common_queue_capacity=20*self.FLAGS.batch_size,
                        common_queue_min=10*self.FLAGS.batch_size,
                        seed=self.rank)
                    [image, label] = data_provider.get(['image', 'label'])
                with tf.name_scope("to-preprocessing"):
                    capacity = 20 * self.FLAGS.batch_size
                    to_pre_queue = data_flow_ops.FIFOQueue(capacity=capacity,
                                                           dtypes=[image.dtype, label.dtype],
                                                           shapes=None,
                                                           name="to_pre_queue")
                    to_pre_op = to_pre_queue.enqueue([image, label])
                    queue_runner.add_queue_runner(queue_runner.QueueRunner(to_pre_queue, [to_pre_op] * Pipeline.QR_THREADS))
                    tf.summary.scalar("to_pre_fraction_of_%d_full" % capacity,
                                    math_ops.to_float(to_pre_queue.size()) * (1. / capacity))
                    image, label = to_pre_queue.dequeue()
                with tf.name_scope("preprocessing"):#TODO
                    image = img_pre_fn(image, self.train_image_size, self.train_image_size, fast_mode=self.FLAGS.fast_mode)
                with tf.name_scope("to-allgather"):
                    capacity = 20 * self.FLAGS.batch_size
                    to_allg_queue = data_flow_ops.FIFOQueue(capacity=capacity,
                                                            dtypes=[image.dtype, label.dtype],
                                                            shapes=[[self.train_image_size, self.train_image_size, 3], []],
                                                            name="to_allgather_queue")#[image.get_shape(), label.get_shape()])
                    queue_runner.add_queue_runner(queue_runner.QueueRunner(to_allg_queue, [to_allg_queue.enqueue([image, label])] * Pipeline.QR_THREADS))
                    tf.summary.scalar("to_allgather_fraction_of_%d_full" % capacity,
                                   math_ops.to_float(to_allg_queue.size()) * (1. / capacity))

                # num_preprocessors = tf.placeholder(tf.int32, shape=[], name="num_preprocessors)
                # self.num_hvd_send_tensor = 
                send_images, send_labels = to_allg_queue.dequeue_many(self.num_hvd_send)
                # if rank == #TODO
                all_images = hvd.allgather(send_images, name="hvd_allgather")
                all_labels = hvd.allgather(send_labels, name="hvd_allgather")
                #TODO: Remove extra queues
                with tf.name_scope("to-compute"):
                    capacity = 30 * self.FLAGS.batch_size
                    to_compute_queue = data_flow_ops.FIFOQueue(capacity=capacity,
                                                               dtypes=[image.dtype, label.dtype],
                                                               shapes=[[self.train_image_size, self.train_image_size, 3], []],#TODO
                                                               name="to_compute_queue")#[image.get_shape(), label.get_shape()])
                    queue_runner.add_queue_runner(queue_runner.QueueRunner(to_compute_queue, [to_compute_queue.enqueue_many([all_images, all_labels])]))#1 thread!
                    tf.summary.scalar("to_compute_fraction_of_%d_full" % capacity,
                                   math_ops.to_float(to_compute_queue.size()) * (1. / capacity))
                image, label = to_compute_queue.dequeue()
        elif self.is_single_bcast:
            ### SINGLE BROADCAST ###
            img_pre_fn = preprocessing_factory.get_preprocessing(self.FLAGS.preprocessing_name, 
                                                                 is_training=True)
            allg_images_name = "allgather-images-op"
            allg_labels_name = "allgather-labels-op"
            bcast_images_name = "bcast-images-op"
            bcast_labels_name = "bcast-labels-op"
            if 0 in self.member_of_group: #If we belong to group 0, initialize the reading and preprocessing pipeline
                with tf.device("/cpu:0"):
                    with tf.name_scope("reading"):
                        data_provider = slim.dataset_data_provider.DatasetDataProvider(
                            self.dataset, num_readers=self.FLAGS.num_data_readers,
                            common_queue_capacity=20*self.FLAGS.batch_size,
                            common_queue_min=10*self.FLAGS.batch_size,
                            seed=self.rank)
                        [image, label] = data_provider.get(['image', 'label'])
                    image, label = create_qr("to-pre", 10 * self.FLAGS.batch_size, [image, label], None, [image.dtype, label.dtype], Pipeline.QR_THREADS, False, False)

                    with tf.name_scope("preprocessing"):
                        image = img_pre_fn(image, self.train_image_size, self.train_image_size, fast_mode=self.FLAGS.fast_mode)

                    send_images, send_labels = create_qr("to-allg", 10 * self.FLAGS.batch_size, [image, label], [[self.train_image_size, self.train_image_size, 3], []], [image.dtype, label.dtype], Pipeline.QR_THREADS, False, True, self.num_hvd_send)
                all_images = hvd.allgather(send_images, group=0, name=allg_images_name)
                all_labels = hvd.allgather(send_labels, group=0, name=allg_labels_name)
                all_images, all_labels = create_qr("to-bcast", 20 * self.FLAGS.batch_size, [all_images, all_labels], [[self.train_image_size, self.train_image_size, 3], []], [post_pre_image_dtype, post_pre_label_dtype], 1, True, True, self.images_per_bcast)
            if 1 in self.member_of_group:
                # For the middle man rank, reset all_images and all_labels
                # names to their broadcasted tensors so that the bcast is
                # performed. Note that the bcast root is rank 0 since the
                # group1 sent to init had this rank listed first, meaning that
                # the resulting mpi group comm has this rank has rank 0
                if len(self.member_of_group) == 1:
                    # Then not middle man, so construct holder variable WITH CORRECT NAME!
                    # tf.Variable(self.num_hvd_send?
                    all_images = tf.zeros([self.images_per_bcast, self.train_image_size, self.train_image_size, 3], dtype=post_pre_image_dtype)
                    all_labels = tf.zeros([self.images_per_bcast]                                       , dtype=post_pre_label_dtype) #shape of [] turns into 1D instead of 0D
                all_images = hvd.broadcast(all_images, 0, group=1, name=bcast_images_name)
                all_labels = hvd.broadcast(all_labels, 0, group=1, name=bcast_labels_name)
            image, label = create_qr("to-compute", 20 * self.FLAGS.batch_size, [all_images, all_labels], [[self.train_image_size, self.train_image_size, 3], []], [post_pre_image_dtype, post_pre_label_dtype], 1, True, False)
        elif self.is_multi_bcast:
            ### MULTIPLE BROADCAST
            # print("Rank:", rank, member_of_group, group_rank_list)
            img_pre_fn = preprocessing_factory.get_preprocessing(self.FLAGS.preprocessing_name, 
                                                                 is_training=True)
            # allg_image_name = "allgathered-image" # need some naming commonalities
            # allg_label_name = "allgathered-label"
            allg_images_name = "allgather-images-op"
            allg_labels_name = "allgather-labels-op"
            bcast_images_name = "bcast-images-op"
            bcast_labels_name = "bcast-labels-op"
            # if 0 in member_of_group: #If we belong to group 0, initialize the reading and preprocessing pipeline
            if self.rank < self.FLAGS.num_pre:
                with tf.device("/cpu:0"):
                    with tf.name_scope("reading"):
                        data_provider = slim.dataset_data_provider.DatasetDataProvider(
                            self.dataset, num_readers=self.FLAGS.num_data_readers,
                            common_queue_capacity=20*self.FLAGS.batch_size,
                            common_queue_min=10*self.FLAGS.batch_size,
                            seed=self.rank)
                        [image, label] = data_provider.get(['image', 'label'])

                    image, label = create_qr("to-pre", 10 * self.FLAGS.batch_size, [image, label], None, [image.dtype, label.dtype], Pipeline.QR_THREADS, False, False)

                    with tf.name_scope("preprocessing"):
                        image = img_pre_fn(image, self.train_image_size, self.train_image_size, fast_mode=self.FLAGS.fast_mode)
                        # image = tf.Print(image, ["using preprocessed image"])
                    send_images, send_labels = create_qr("to-bcast", 20 * self.FLAGS.batch_size, [image, label], [[self.train_image_size, self.train_image_size, 3], []], [image.dtype, label.dtype], 2 * Pipeline.QR_THREADS, False, True, self.images_per_bcast)
            else:
                send_images = tf.zeros([self.images_per_bcast, self.train_image_size, self.train_image_size, 3], dtype=post_pre_image_dtype)
                send_labels = tf.zeros([self.images_per_bcast]                                                 , dtype=post_pre_label_dtype)
            with tf.device("/cpu:0"):
                bcast_images_root = "broadcast-images-"
                bcast_labels_root = "broadcast-labels-"
                bcast_images_per_group = [hvd.broadcast(send_images, i, group=i, name=bcast_images_root + str(i)) for i in range(self.FLAGS.num_pre)]
                bcast_labels_per_group = [hvd.broadcast(send_labels, i, group=i, name=bcast_labels_root + str(i)) for i in range(self.FLAGS.num_pre)]
                
                with tf.name_scope("to-compute"):
                    capacity = 30 * self.FLAGS.batch_size
                    to_compute_q = data_flow_ops.FIFOQueue(capacity=capacity,
                                                    dtypes=[post_pre_image_dtype, post_pre_label_dtype],
                                                    shapes=[[self.train_image_size, self.train_image_size, 3], []], 
                                                    name="to-compute-queue")
                    to_comp_ops = [to_compute_q.enqueue_many([bcast_images_per_group[i], bcast_labels_per_group[i]]) for i in range(self.FLAGS.num_pre)]
                    queue_runner.add_queue_runner(queue_runner.QueueRunner(to_compute_q, to_comp_ops))
                    tf.summary.scalar("to_compute_fraction_of_%d_full" % capacity,
                                      math_ops.to_float(to_compute_q.size()) * (1. / capacity))
                    image, label = to_compute_q.dequeue()
        return image, label
    
    def get_preprocessing_eater(self):
        image, label = self.get_image_labels()
        batch_out = tf.train.batch(
            [image, label],
            batch_size=self.FLAGS.batch_size,
            num_threads=self.FLAGS.num_batching_threads,
            capacity=self.FLAGS.batch_queue_size * self.FLAGS.batch_size,
            shapes=[image.get_shape(), []])
        with tf.control_dependencies(batch_out):
            eater = tf.no_op()
        return eater

    def compute_loss(self):
        image, label = self.get_image_labels()
        with tf.device("/device:GPU:0"):
            with tf.name_scope("batching"):
                images, labels = tf.train.batch(
                    [image, label],
                    batch_size=FLAGS.batch_size,
                    num_threads=FLAGS.num_batching_threads,
                    capacity=FLAGS.batch_queue_size * FLAGS.batch_size,
                    shapes=[image.get_shape(), []])
                labels = slim.one_hot_encoding(labels, self.dataset.num_classes)

            # summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
            logits, end_points = self.network_fn(images)#, reuse=gpu_idx!=0)

            correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            tf.summary.scalar("train-accuracy", acc)

            tf.losses.softmax_cross_entropy(labels, logits)
            losses = tf.get_collection(tf.GraphKeys.LOSSES, None)# not sure None is necessary
            #Ignoring other types of losses...
            loss = tf.add_n(losses, name="loss")
            tf.summary.scalar("loss", loss)
        return loss
    
    def get_train_dir(self):
        return self.train_dir

    def get_rank(self):
        return self.rank
