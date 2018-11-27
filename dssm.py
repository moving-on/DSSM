import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib import learn
from tensorflow.contrib.lookup.lookup_ops import MutableHashTable
from tensorflow.python.framework import constant_op

UNK_ID = 1
_START_VOCAB = ['_PAD', '_UNK']

tf.flags.DEFINE_string("query_file", "./data/sample_q.txt", "query file")
tf.flags.DEFINE_string("doc_file", "./data/sample_d.txt", "doc file")
tf.flags.DEFINE_integer("batch_size", 20, "batch_size")
tf.flags.DEFINE_integer("num_epochs", 100, "number of epoch")
tf.flags.DEFINE_integer("embedding_dim", 100, "demention of embedding")
tf.flags.DEFINE_integer("vocab_size", 10000, "vocab size")
tf.flags.DEFINE_integer("neg_num", 4, "neg num")

FLAGS = tf.flags.FLAGS

class DSSM(object):
    def __init__(self, sess, embedding_dim, vocab_size, neg_num=4):
        self.sess = sess
        self.word2index = MutableHashTable(
                key_dtype=tf.string,
                value_dtype=tf.int64,
                default_value=UNK_ID,
                shared_name="in_table",
                name="in_table",
                checkpoint=True)
        self.queries = tf.placeholder(name="query", shape=[None, None], dtype=tf.string)
        self.docs = tf.placeholder(name="doc", shape=[neg_num+1, None, None], dtype=tf.string)
        self.query_len = tf.placeholder(name="query_len", shape=[None, 1], dtype=tf.float32)
        self.doc_len = tf.placeholder(name="doc_len", shape=[neg_num+1, None, 1], dtype=tf.float32)
        self.query_ids = self.word2index.lookup(self.queries)
        self.doc_ids = [ self.word2index.lookup(doc) for doc in tf.unstack(self.docs) ]

        embedding = tf.get_variable(name="embedding", dtype=tf.float32, 
                shape=[vocab_size, embedding_dim],
                initializer=tf.random_normal_initializer(0, 0.1))
        self.embedding = tf.concat((tf.zeros(shape=[2, embedding_dim]),
                embedding[2:, :]), 0)

        self.query_emb = tf.div(tf.reduce_sum(tf.nn.embedding_lookup(self.embedding, self.query_ids), axis=1), self.query_len)
        self.doc_emb = [ tf.div(tf.reduce_sum(tf.nn.embedding_lookup(self.embedding, self.doc_ids[i]), axis=1), self.doc_len[i])
                                 for i in range(neg_num+1) ]

        def feed_forward(input_emb, scope="fnn", reuse=False):
            with tf.variable_scope(scope, reuse=reuse):
                hidden1 = tf.layers.dense(input_emb, units=45, activation="tanh", name="dense1")
                out = tf.layers.dense(hidden1, units=30, activation=None, name="dense2")
                return out

        self.query_out = feed_forward(self.query_emb)
        self.doc_out = [ feed_forward(doc_emb, reuse=True) for doc_emb in self.doc_emb ]
        self.query_norm = tf.sqrt(tf.reduce_sum(tf.square(self.query_out), axis=1))
        self.doc_norm = [ tf.sqrt(tf.reduce_sum(tf.square(doc_out), axis=1)) for doc_out in self.doc_out ]
        self.ori_prob = [ tf.reduce_sum(tf.multiply(self.query_out, self.doc_out[i]), axis=1) for i in range(neg_num + 1) ]
        self.ori_sim = [ self.ori_prob[i] / (self.query_norm * self.doc_norm[i]) for i in range(neg_num + 1) ]

        self.gamma = tf.Variable(initial_value=1.0, expected_shape=[], dtype=tf.float32, trainable=True)
        self.sim = tf.convert_to_tensor(self.ori_sim) * self.gamma

        self.prob = tf.nn.softmax(self.sim, dim=0)
        self.pos_prob = self.prob[0]
        self.loss = -tf.reduce_mean(tf.log(self.pos_prob))
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(self.loss)

    def train(self, queries, docs, query_len, doc_len):
        _, loss, pos_prob, prob, emb, doc_emb, query_emb, query_ids, doc_ids, sim, query_out, doc_out= self.sess.run([self.train_op, self.loss, self.pos_prob, self.prob, self.embedding, self.doc_emb, self.query_emb, self.query_ids, self.doc_ids, self.sim, self.query_out, self.doc_out], 
                          feed_dict={self.queries:queries, self.docs:docs,
                                     self.query_len:query_len, self.doc_len:doc_len})
        #print query_ids
        #print doc_ids
        #print query_out
        #print doc_out[0]
        #print sim
        #print prob
        #print pos_prob
        return loss, pos_prob, prob

def load_data(filename):
    cropus = []
    cropus_len = []
    max_len = 0
    with open(filename, "r") as infile:
        for line in infile:
            cropus.append(line.strip().split(" "))
            cropus_len.append(len(line.strip().split(" ")))
            if max_len < len(line.strip().split(" ")):
                max_len = len(line.strip().split(" "))
    return cropus,cropus_len,  max_len

def build_vocabulary(cropus):
    terms = set()
    for line in cropus:
        for term in line:
            terms.add(term)
    vocab = _START_VOCAB + list(terms)
    FLAGS.vocab_size = len(vocab)
    return vocab

def padding(cropus, max_len):
    for i in range(len(cropus)):
        cropus[i] += [_START_VOCAB[0]] * (max_len - len(cropus[i]))
    return cropus

def sample_neg(cropus, cropus_len):
    cropus = np.array(cropus)
    cropus_len = np.array(cropus_len)
    data = np.repeat(cropus, [FLAGS.neg_num + 1], axis=0)
    data_len = np.repeat(cropus_len, [FLAGS.neg_num + 1], axis=0)
    data_size = data.shape[0]
    cropus_size = cropus.shape[0]
    for i in range(0, data_size, FLAGS.neg_num+1):
        random_indices = np.random.randint(1, cropus_size, FLAGS.neg_num)
        data[i+1:i+FLAGS.neg_num+1] = cropus[random_indices]
        data_len[i+1:i+FLAGS.neg_num+1] = cropus_len[random_indices]
    return data, data_len

def preprocess():
    query_cropus, query_len, max_query_len = load_data(FLAGS.query_file)
    doc_cropus, doc_len, max_doc_len = load_data(FLAGS.doc_file) 
    vocab = build_vocabulary(query_cropus + doc_cropus)
    query_cropus = padding(query_cropus, max_query_len)
    doc_cropus = padding(doc_cropus, max_doc_len)
    doc_cropus, doc_len = sample_neg(doc_cropus, doc_len)
    return vocab, query_cropus, doc_cropus, query_len, doc_len

def gen_batch(query, doc, query_len, doc_len):
    query = np.array(query)
    doc = np.array(doc)
    query_len = np.array(query_len)
    doc_len = np.array(doc_len)
    data_size = query.shape[0]
    batch_num = query.shape[0] / FLAGS.batch_size
    for epoch in range(FLAGS.num_epochs):
        for i in range(batch_num):
            start = i * FLAGS.batch_size
            end = (i+1) * FLAGS.batch_size
            batch_query = query[start:end, :]
            batch_query_len = query_len[start:end]
            batch_query_len = np.reshape(batch_query_len, (FLAGS.batch_size, -1))
            batch_doc = doc[start*(FLAGS.neg_num+1):end*(FLAGS.neg_num+1)]
            batch_doc = np.reshape(batch_doc, (FLAGS.batch_size, FLAGS.neg_num+1, -1))
            batch_doc = np.transpose(batch_doc, (1,0,2))
            batch_doc_len = doc_len[start*(FLAGS.neg_num+1):end*(FLAGS.neg_num+1)]
            batch_doc_len = np.reshape(batch_doc_len, (FLAGS.batch_size, FLAGS.neg_num+1, -1))
            batch_doc_len = np.transpose(batch_doc_len, (1,0,2))
            yield batch_query, batch_doc, batch_query_len, batch_doc_len
    
def main(argv=None):
    vocab, query_cropus, doc_cropus, query_len, doc_len = preprocess()
    with tf.Session() as sess:
        model = DSSM(sess=sess, embedding_dim=FLAGS.embedding_dim, vocab_size=FLAGS.vocab_size, neg_num=FLAGS.neg_num)
        op_in = model.word2index.insert(constant_op.constant(vocab),
                constant_op.constant(list(range(FLAGS.vocab_size)), dtype=tf.int64))
        sess.run(op_in)
        sess.run(tf.global_variables_initializer())
        print tf.trainable_variables()
        #pad_str = tf.placeholder(dtype=tf.string, name="pad_str", shape=[])
        #pad_id = model.word2index.lookup(pad_str)
        #print(sess.run(pad_id, feed_dict={pad_str:_START_VOCAB[0]}))
        #print(sess.run(pad_id, feed_dict={pad_str:_START_VOCAB[1]}))
        #print len(vocab)
        batch_iter = gen_batch(query_cropus, doc_cropus, query_len, doc_len)
        for batch_query, batch_doc, batch_query_len, batch_doc_len in batch_iter:
            #print batch_query
            #print batch_doc[0]
            #print batch_doc[1]
            #print ''.join(batch_doc[1][0])
            #print ''.join(batch_doc[2][0])
            #print ''.join(batch_doc[3][0])
            #print ''.join(batch_doc[4][0])
            #query_out = sess.run([model.query_emb], feed_dict={model.queries:batch_query, model.docs:batch_doc,
                                                               #model.query_len:batch_query_len, model.doc_len:batch_doc_len})
            #print query_out
            loss, pos_prob, prob = model.train(batch_query, batch_doc, batch_query_len, batch_doc_len)
            #print prob
            #print pos_prob
            print "loss={}".format(loss) 

if __name__ == "__main__":
    tf.app.run()
