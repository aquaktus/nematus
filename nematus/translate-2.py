#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Translates a source file using a translation model.
'''
import argparse
import logging
import numpy
import sys
import time

import tensorflow as tf

from threading import Thread
from multiprocessing import Queue
from collections import defaultdict

from model import StandardModel
from util import load_config, seq2words, prepare_data
from compat import fill_options
from settings import TranslationSettings
from nmt import init_or_restore_variables, load_dictionaries, read_all_lines
import exception


def translate(sess, model, config, input_file, output_file=sys.stdin,
              batch_size=80):
    start_time = time.time()
    _, _, _, num_to_target = load_dictionaries(config)
    logging.info("NOTE: Length of translations is capped to {}".format(config.translation_maxlen))

    n_sent = 0
    try:
        batches, idxs = read_all_lines(config, input_file.readlines(),
                                       batch_size)
    except exception.Error as x:
        logging.error(x.msg)
        sys.exit(1)
    in_queue, out_queue = Queue(), Queue()
    model._get_beam_search_outputs(config.beam_size)

    def translate_worker(in_queue, out_queue, model, sess, config):
        while True:
            job = in_queue.get()
            if job is None:
                break
            idx, x = job
            y_dummy = numpy.zeros(shape=(len(x),1))
            x, x_mask, _, _ = prepare_data(x, y_dummy, config.factors, maxlen=None)
            try:
                samples = model.beam_search(sess, x, x_mask, config.beam_size)
                out_queue.put((idx, samples))
            except:
                in_queue.put(job)

    threads = [None] * config.n_threads
    for i in xrange(config.n_threads):
        threads[i] = Thread(
                        target=translate_worker,
                        args=(in_queue, out_queue, model, sess, config))
        threads[i].deamon = True
        threads[i].start()

    for i, batch in enumerate(batches):
        in_queue.put((i,batch))
    outputs = [None]*len(batches)
    for _ in range(len(batches)):
        i, samples = out_queue.get()
        outputs[i] = list(samples)
        n_sent += len(samples)
        logging.info('Translated {} sents'.format(n_sent))
    for _ in range(config.n_threads):
        in_queue.put(None)
    outputs = [beam for batch in outputs for beam in batch]
    outputs = numpy.array(outputs, dtype=numpy.object)
    outputs = outputs[idxs.argsort()]

    for beam in outputs:
        if config.normalize:
            beam = map(lambda (sent, cost): (sent, cost/len(sent)), beam)
        beam = sorted(beam, key=lambda (sent, cost): cost)
        if config.n_best:
            for sent, cost in beam:
                line = "{} [{}]\n".format(seq2words(sent, num_to_target), cost)
                output_file.write(line)
        else:
            best_hypo, cost = beam[0]
            line = seq2words(best_hypo, num_to_target) + '\n'
            output_file.write(line)
    duration = time.time() - start_time
    logging.info('Translated {} sents in {} sec. Speed {} sents/sec'.format(n_sent, duration, n_sent/duration))

class Translation(object):
    """
    Models a translated segment.
    """
    def __init__(self, source_words, target_words, sentence_id=None, score=0, hypothesis_id=None):
        self.source_words = source_words
        self.target_words = target_words
        self.sentence_id = sentence_id
        self.score = score
        self.hypothesis_id = hypothesis_id


class QueueItem(object):
    """
    Models items in a queue.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Translator(object):
    def __init__(self, settings):
        """
        Loads translation models.
        """
        self._models = settings.models
        self._num_processes = settings.num_processes
        self._verbose = settings.verbose
        self._retrieved_translations = defaultdict(dict)
        self._batch_size = settings.b

        # load model options
        self._load_model_options()

        tf_config = tf.ConfigProto()
        tf_config.allow_soft_placement = True
        self.sess = tf.Session(config=tf_config)
        self.models = self._load_models(self.sess)

    def _load_model_options(self):
        """
        Loads config options for each model.
        """

        self._options = []
        for model in self._models:
            config = load_config(model)
            # backward compatibility
            fill_options(config)
            config['reload'] = model
            self._options.append(argparse.Namespace(**config))

        _, _, _, self._num_to_target = load_dictionaries(self._options[0])

    def _load_models(self, sess):
        """
        Loads models and returns them
        """
        logging.debug("Loading models\n")

        models = []
        for i, options in enumerate(self._options):
            with tf.variable_scope("model%d" % i) as scope:
                model = StandardModel(options)
                saver = init_or_restore_variables(options, sess,
                                                  ensemble_scope=scope)
                models.append(model)

        logging.info("NOTE: Length of translations is capped to {}".format(self._options[0].translation_maxlen))
        return models

    def translate(self, input_file, output_file):
        """
        """
        translate(self.sess, self.models[0], self._options[0], input_file,
                  output_file, self._batch_size)


def main(input_file, output_file, translation_settings):
    """
    Translates a source language file (or STDIN) into a target language file
    (or STDOUT).
    """
    translator = Translator(translation_settings)
    translations = translator.translate(input_file, output_file)
    logging.info('Done')


if __name__ == "__main__":
    # parse console arguments
    translation_settings = TranslationSettings(from_console_arguments=True)
    input_file = translation_settings.input
    output_file = translation_settings.output
    # start logging
    level = logging.DEBUG if translation_settings.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')
    main(input_file, output_file, translation_settings)
