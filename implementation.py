import tensorflow as tf
import numpy as np
import glob #this will be useful when reading reviews from file
import os
import tarfile
import string


batch_size = 50

def load_data(glove_dict):
    """
    Take reviews from text files, vectorize them, and load them into a
    numpy array. Any preprocessing of the reviews should occur here. The first
    12500 reviews in the array should be the positive reviews, the 2nd 12500
    reviews should be the negative reviews.
    RETURN: numpy array of data with each row being a review in vectorized
    form"""
    filename = 'reviews.tar.gz'
    seq_length = 40
    matrix = []
    statinfo = os.stat(filename)
    with tarfile.open(filename, "r") as tarball:
        dir = os.path.dirname(__file__)
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tarball, os.path.join(dir,"data2/"))
    
    dir = os.path.dirname(__file__)
    file_list = glob.glob(os.path.join(dir,'data2/pos/*'))
    file_list.extend(glob.glob(os.path.join(dir,'data2/neg/*')))
    for f in file_list:
        with open(f, "r") as openf:
            s = openf.read()
            no_punct = ''.join(c for c in s if c not in string.punctuation)
            no_punct = no_punct.lower()
            row = []
            for idx in range(0,seq_length):
                words = no_punct.split()
                if idx < len(words):
                    word = words[idx]
                else:
                    word = "UNK"
                
                if word in glove_dict:
                    row.append(glove_dict[word])
                else:
                    row.append(0)
                matrix.append(row)
    
    data = np.array(matrix)

    return data


def load_glove_embeddings():
    """
    Load the glove embeddings into a array and a dictionary with words as
    keys and their associated index as the value. Assumes the glove
    embeddings are located in the same directory and named "glove.6B.50d.txt"
    RETURN: embeddings: the array containing word vectors
            word_index_dict: a dictionary matching a word in string form to
            its index in the embeddings array. e.g. {"apple": 119"}
    """
    data = open("glove.6B.50d.txt",'r',encoding="utf-8")
    #if you are running on the CSE machines, you can load the glove data from here
    #data = open("/home/cs9444/public_html/17s2/hw2/glove.6B.50d.txt",'r',encoding="utf-8")
    embedding = []
    word_index_dict = {"UNK": 0}
    count = 0
    for line in data:
        value = line.split()
        word = value[0]
        del value[0]
        row = []
        for num in value:
            row.append(np.float32(num))
        embedding.append(row)
        word_index_dict.update({word:count})
        count = count + 1

    embeddings = np.array(embedding)
    return embeddings, word_index_dict


def define_graph(glove_embeddings_arr):
    """
    Define the tensorflow graph that forms your model. You must use at least
    one recurrent unit. The input placeholder should be of size [batch_size,
    40] as we are restricting each review to it's first 40 words. The
    following naming convention must be used:
        Input placeholder: name="input_data"
        labels placeholder: name="labels"
        accuracy tensor: name="accuracy"
        loss tensor: name="loss"

    RETURN: input placeholder, labels placeholder, dropout_keep_prob, optimizer, accuracy and loss
    tensors"""
    
    tf.reset_default_graph()
    batch_size = 50
    num_classes = 2
    vector_dim = 50
    seq_length = 40
    num_layers = 3
    lstm_size = 64
    learning_rate=1e-5;

    input_data = tf.placeholder(tf.int32, shape=[batch_size, seq_length])
    labels = tf.placeholder(tf.int32, shape=[batch_size, num_classes])
    dropout_keep_prob = tf.placeholder_with_default(0.5, shape=())

    # word vector of the input data
    data = tf.Variable(tf.zeros([batch_size, seq_length, vector_dim]),dtype=tf.float32)
    data = tf.nn.embedding_lookup(glove_embeddings_arr, input_data)

    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=dropout_keep_prob)
    value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

    weight = tf.Variable(tf.truncated_normal([lstm_size, num_classes]))
    bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
    value = tf.transpose(value, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)

    correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


    return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss
