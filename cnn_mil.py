#!/usr/bin/env python3

"""
Author: Dan Salo
Initial Commit: 12/1/2016

Purpose: Implement Convolutional Multiple Instance Learning for distributed learning over MNIST dataset
"""

import sys
sys.path.append('../')

from tensorbase.base import Model
from tensorbase.base import Layers
from tensorbase.data import Mnist
from data import CancerData

import tensorflow as tf
import numpy as np


# Global Dictionary of Flags
flags = {
    'save_directory': 'summaries/',
    'model_directory': 'conv_mil1/',
    'restore': False,
    'restore_file': 'part_4.ckpt.meta',
    'datasets': 'CancerData',
    'num_classes': 2,
    'class_threshold': 2.5,	# Threshold for binary classification
    'image_dim': 2560,
    'batch_size': 10,
    'display_step': 500,
    'weight_decay': 1e-7,
    'lr_decay': 0.999,
    'learn_rate': 0.001,
    "path_to_image_directory": "/scratch2/mammosprint/",
    "path_to_metadata": "metadata.json",
    "gpu": 1
}


class ConvMil(Model):
    def __init__(self, flags_input, run_num, restore):
        self._set_placeholders()
        super().__init__(flags_input, run_num, restore=restore)
        self.print_log("Seed: %d" % flags['seed'])
        self.valid_results = list()
        self.test_results = list()
        self.learn_rate = flags_input['learn_rate']

    def _data(self):
        self.data = CancerData(self.flags)
        self.num_train_images = self.data.num_train_images
        self.num_valid_images = self.data.num_valid_images
        self.num_test_images = self.data.num_test_images
        print(self.num_train_images, self.num_valid_images, self.num_test_images)

    def _set_placeholders(self):
        self.x = tf.placeholder(tf.float32, [None, flags['image_dim'], flags['image_dim'], 1], name='x')
        self.y = tf.placeholder(tf.int32, shape=[None, flags['num_classes']], name='y')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

    def _summaries(self):
        tf.summary.scalar("Total Loss", self.cost)
        #tf.scalar_summary("Cross Entropy Loss", self.xentropy)
        #tf.scalar_summary("Weight Decay Loss", self.weight)
        #tf.image_summary("x", self.x)

    def _network(self):
        with tf.variable_scope("model"):
            net = Layers(self.x)
            #net.avgpool()
            net.conv2d(3, 16, stride=2)
            net.conv2d(3, 32, stride=2)
            net.maxpool()
            net.conv2d(5, 32, stride=2)
            net.maxpool()
            net.conv2d(3, 64)
            net.maxpool()
            net.conv2d(3, 64)
            net.maxpool()
            net.conv2d(1, 500)
            net.conv2d(1, self.flags['num_classes'], activation_fn=tf.nn.sigmoid)
            self.prior = net.get_output()
            net.noisy_and(self.flags['num_classes'])
            #net.fc(self.flags['num_classes'])
            self.y_hat = net.get_output()
            self.logits = self.y_hat

    def _optimizer(self):
        #shape = [self.flags['batch_size'], self.flags['num_classes']]
        #y_hat = tf.to_float(self.y_hat) + tf.constant(0.00001, shape=shape)
        
        #cross_entropy = -tf.reduce_sum(tf.to_float(self.y) * tf.log(softmax), reduction_indices=[1])
        #self.xentropy = const * tf.reduce_sum(cross_entropy)

        const = 1./self.flags['batch_size']
        self.xentropy = const * tf.nn.softmax_cross_entropy_with_logits(logits=tf.to_float(self.y_hat),
                                                                        labels=tf.to_float(self.y),
                                                                        name='xentropy')
        self.weight = self.flags['weight_decay'] * tf.add_n(tf.get_collection('weight_losses'))
        self.cost = tf.reduce_sum(self.xentropy + self.weight)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost)

    def _generate_train_batch(self):
        self.train_batch_y, train_batch_x = self.data.next_train_batch(self.flags['batch_size'])
        self.train_batch_x = np.reshape(train_batch_x, [self.flags['batch_size'],
                                        self.flags['image_dim'], self.flags['image_dim'], 1])
        im = self.train_batch_x[0]

    def _generate_valid_batch(self):
        self.valid_batch_y, valid_batch_x, valid_number, batch_size = self.data.next_valid_batch(self.flags['batch_size'])
        self.valid_batch_x = np.reshape(valid_batch_x, [batch_size, self.flags['image_dim'], self.flags['image_dim'], 1])
        return valid_number

    def _generate_test_batch(self):
        self.test_batch_y, test_batch_x, test_number, batch_size = self.data.next_test_batch(self.flags['batch_size'])
        self.test_batch_x = np.reshape(test_batch_x, [batch_size, self.flags['image_dim'], self.flags['image_dim'], 1])

        return test_number

    def _run_train_iter(self):
        rate = self.learn_rate * self.flags['lr_decay']
        self.summary, _ = self.sess.run([self.merged, self.optimizer],
                                        feed_dict={self.x: self.train_batch_x, self.y: self.train_batch_y,
                                                   self.lr: rate})

    def _evaluate(self, labels, logits):
        preds = np.reshape(logits, [-1, self.flags['num_classes']])
        A = np.argmax(labels, 1)
        B = np.argmax(preds, 1)
        correct_prediction = np.equal(A, B)

        return correct_prediction

    def _run_train_summary_iter(self):
        rate = self.learn_rate * self.flags['lr_decay']
        w_1 = tf.get_collection('weight_losses')[0]
        w_8 = tf.get_collection('weight_losses')[-1]
        self.summary, self.loss, logits, w1_loss, w8_loss, _ = self.sess.run(
                [self.merged, self.cost, self.logits, w_1, w_8, self.optimizer],
                feed_dict={self.x: self.train_batch_x, self.y: self.train_batch_y,
                           self.lr: rate})

        print('Loss:', w1_loss, w8_loss)
        if np.isnan(w1_loss):
            print("Loss on 1st layer is NaN")
            assert(np.isnan(w1_loss) == false)
        if np.isnan(self.loss):
            print("Loss function is NaN")
            assert(np.isnan(self.loss) == false)

        #correct_prediction = self._evaluate(self.train_batch_y, logits)
        #accuracy = np.mean(correct_prediction)
        #self.print_log("Accuracy on Training batch: %f" % accuracy)

    def _run_train_iter(self):
        logits = self.sess.run([self.logits], feed_dict={self.x: self.train_batch_x})

        return self._evaluate(self.train_batch_y, logits)

    def _run_valid_iter(self):
        logits = self.sess.run([self.logits], feed_dict={self.x: self.valid_batch_x})

        correct_prediction = self._evaluate(self.valid_batch_y, logits) 
        self.valid_results = np.concatenate((self.valid_results, correct_prediction))

    def _run_test_iter(self):
        logits = self.sess.run([self.logits], feed_dict={self.x: self.test_batch_x})

        correct_prediction = self._evaluate(self.test_batch_y, logits)
        self.test_results = np.concatenate((self.test_results, correct_prediction))

    def _record_train_metrics(self):
        self.print_log("Batch Number: " + str(self.step) + ", Total Loss= " + "{:.6f}".format(self.loss) + "\n")

    def _record_valid_metrics(self):
        accuracy = np.mean(self.valid_results)
        self.print_log("Accuracy on Validation Set: %f" % accuracy)
        file = open(self.flags['restore_directory'] + 'ValidAccuracy.txt', 'w')
        file.write('Valid set accuracy:')
        file.write(str(accuracy))
        file.close()

    def _record_test_metrics(self):
        accuracy = np.mean(self.test_results)
        self.print_log("\nAccuracy on Test Set: %f\n" % accuracy)
        file = open(self.flags['restore_directory'] + 'TestAccuracy.txt', 'w')
        file.write('Test set accuracy:')
        file.write(str(accuracy))
        file.close()

    def run(self):
        epochs = 40
        train_iters = int(self.data._num_train_images / self.flags['batch_size'])
        valid_iters = int(self.data._num_valid_images / self.flags['batch_size'])
        test_iters = int(self.data._num_test_images / self.flags['batch_size'])

        for j in range(epochs):
            for i in range(train_iters):
                self._generate_train_batch()
                self._run_train_summary_iter()
                self._record_train_metrics()
                self.step += 1

            print("Number of epochs:", self.data.train_epochs_completed)

            #results = list()
            #for i in range(train_iters):
            #    self._generate_train_batch()
            #    curr = self._run_train_iter()
            #    results = np.concatenate((results, curr))
            #accuracy = np.mean(results)
            #self.print_log("\nAccuracy on Training Set: %f\n" % accuracy)

            for i in range(valid_iters):
                self._generate_valid_batch()
                self._run_valid_iter()
            self._record_valid_metrics()

            for i in range(test_iters):
                self._generate_test_batch()
                self._run_test_iter()
            self._record_test_metrics()

            self.print_log("EPOCH %d DONE!\n\n\n" % (j+1))
            self._save_model(0)

            self.data.index_in_valid_epoch = -1
            self.data.index_in_test_epoch = -1
            self.valid_results = list()
            self.test_results = list()

        self._save_model(1)

def main():
    flags['seed'] = np.random.randint(1, 1000, 1)[0]
    model_mil = ConvMil(flags, run_num=1, restore=1)
    model_mil.run()


if __name__ == "__main__":
    main()
