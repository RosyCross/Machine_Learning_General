# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 23:57:58 2019


"""

import tensorflow as tf
import numpy as np

"""
must be used if you want to rerun with IDE
"""
tf.reset_default_graph()

"""
Auxiliary functions
"""
num_points = 4;
ref_box_width = 8
ref_box = np.array([[0,0],[0,ref_box_width],[ref_box_width,ref_box_width],[ref_box_width,0]])
ll,ul,ur,lr = ref_box
#ref_box_edges = np.array([0,ref_box_width],[ref_box_width,0],[0,-ref_box_width],[-ref_box_width,0])
ref_box_edges = np.empty((0,2,2), dtype=int)
for idx, idy in zip(range(0,num_points,1),range(1,num_points+1,1)):
    print("idx {}, idy {}".format(idx,idy))
    ref_box_edges = np.append(ref_box_edges, [[ref_box[idx % num_points], ref_box[idy % num_points]]], axis=0 )

min_spacing = 9
x_ref_max = 5
y_ref_max = 5
x_fake_max = 30
y_fake_max = 30
x_zoom_factor = x_fake_max / x_ref_max
y_zoom_factor = y_fake_max / y_ref_max

def genBatchTrainData(dist_start=-40, dist_end=40, rate=0.05):
    #ll,ul,ur,lr = ref_box;    
    train_x = []
    train_y = []
    #for point_on_edge in [ll, ul, ll/2 + ul/2, ur, lr, ur/2+lr/2]:        
    for edge in ref_box_edges:
        edge_vec = edge[1]-edge[0];
        ref_points_on_edge = [edge[0], edge[1], edge[0]/2 + edge[1]/2];
        for point_on_edge in ref_points_on_edge:
            #for distance in np.arange(-50,50,0.1):
            for distance in np.arange(dist_start,dist_end,rate):                
                ##anchor_point = point_on_edge + np.array([distance,0]);
                tmp_vec = np.array([distance,0]) if edge[0][0] == edge[1][0] else np.array([0,distance])
                anchor_point = point_on_edge + tmp_vec;
                tmp_vec = anchor_point - edge[0];
                is_outside = ( edge_vec[0]*tmp_vec[1] - edge_vec[1]*tmp_vec[0]) > 0
                if is_outside:
                    is_drc_vio = abs(distance) < min_spacing
                #    print("outside, {} -- {} --- dist:{}".format(edge, anchor_point, distance))
                else:
                    is_drc_vio = abs(distance) < (min_spacing + ref_box_width)                    
                #    print("not outside, {} -- {} --- dist:{}".format( edge, anchor_point, distance))
                                
                one_shot = []
                one_shot.append(anchor_point - ll) 
                one_shot.append(anchor_point - ul);                            
                one_shot.append(anchor_point - ur);                
                one_shot.append(anchor_point - lr);
                train_y.append(int(is_drc_vio))     
                train_x.append(one_shot)
   
    return train_x, train_y


"""
construct training , test data
"""
x_train, y_train = genBatchTrainData();
shuffled_ix = np.random.permutation(np.arange(len(x_train)))        
x_train = np.array(x_train)[shuffled_ix]
y_train = np.array(y_train)[shuffled_ix]

x_cutoff = int(len(x_train)*0.8)
if len(x_train) - x_cutoff < 10:
    x_cutoff = len(x_train) - 10
x_train, x_test = x_train[:x_cutoff], x_train[x_cutoff:]
y_train, y_test = y_train[:x_cutoff], y_train[x_cutoff:]

"""
must be trained point, try to increase the steepness at the boundary of min spacing
"""
x_must, y_must = genBatchTrainData(dist_start=-min_spacing, dist_end=-min_spacing+1, rate=1)
x_must = np.array(x_must)
y_must = np.array(y_must)
#x_train = np.vstack((x_train, x_must))
#y_train = np.vstack((y_train, y_must))
x_train = np.concatenate((x_train, x_must))
y_train = np.concatenate((y_train, y_must))

x_must, y_must = genBatchTrainData(dist_start=min_spacing, dist_end=min_spacing+1, rate=1)
x_must = np.array(x_must)
y_must = np.array(y_must)
#np.vstack([x_train, x_must])
#np.vstack([y_train, y_must])
x_train = np.concatenate((x_train, x_must))
y_train = np.concatenate((y_train, y_must))


"""
Start the section of computing graph
"""
batch_size = 10;
input_size = 2;
sequence_length = 4;
output_size = 1;
rnn_size = 4; #rnn_states
#dropout_keep_prob =0.5
learning_rate = 0.0005

###############################
x_data   = tf.placeholder(tf.float32, [None, sequence_length, input_size])
y_output = tf.placeholder(tf.int32, [None])
dropout_keep_prob = tf.placeholder(tf.float32)
#h0 = cell.zero_state(32, np.float32)

# Define the RNN cell
cell   = tf.nn.rnn_cell.BasicRNNCell(num_units = rnn_size)
print("cell state_size: ",cell.state_size)

output, state = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
#output, state = tf.nn.dynamic_rnn(cell, x_data, initial_state=h0)
# parameters are variables, waiting for constant later.
output = tf.nn.dropout(output, dropout_keep_prob)

# Get output of RNN sequence
output = tf.transpose(output, [1, 0, 2])
print("output shape: ", output.get_shape())
last   = tf.gather(output, int(output.get_shape()[0]) - 1)
print("last shape: ", last.get_shape())

###### Variables
weight = tf.Variable(tf.truncated_normal([rnn_size, 2], stddev=0.1))
print("weight shape: ", weight.get_shape())
bias   = tf.Variable(tf.constant(0.1, shape=[2]))
logits_out = tf.nn.softmax(tf.matmul(last, weight) + bias)
print("logits_out shape: ", logits_out.get_shape())
#my_test_output = tf.matmul(last, weight) + bias

########## Loss function
# logits=float32, labels=int32
#tf.nn.sigmoid_cross_entropy_with_logits(logits, targets, name=None)
#losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_out, labels=y_output)
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_out, labels=y_output) 
loss   = tf.reduce_mean(losses)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits_out, 1), tf.cast(y_output, tf.int64)), tf.float32))

optimizer  = tf.train.RMSPropOptimizer(learning_rate)
train_step = optimizer.minimize(loss)
#optimizer = tf.train.GradientDescentOptimizer(0.005)
#train_step = optimizer.minimize(loss)

###############################################################################
###############################################################################
epochs = 100
init = tf.initialize_all_variables()

train_loss     = []
test_loss      = []
train_accuracy = []
test_accuracy  = []

# Start training
#sess = tf.Session()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        # Shuffle training data
###        shuffled_ix = np.random.permutation(np.arange(len(x_train)))        
        # Sort x_train and y_train based on shuffled_ix
###        x_train = np.array(x_train)[shuffled_ix]
###        y_train = np.array(y_train)[shuffled_ix]
        
        num_batches = int(len(x_train)/batch_size) + 1
        # TO DO CALCULATE GENERATIONS ExACTLY
        # For each batch.
        for i in range(num_batches):
            # Select train data            
            min_ix = i * batch_size
            max_ix = np.min([len(x_train), ((i+1) * batch_size)])
            #print("==> ", i, " ", batch_size)
            #print("min:", min_ix, "max: ", max_ix, "num_batches:", num_batches);
            x_train_batch = x_train[min_ix:max_ix]
            y_train_batch = y_train[min_ix:max_ix]
            if min_ix >= max_ix:
                continue;
            # Run train step
            train_dict = {x_data:x_train_batch, y_output:y_train_batch, dropout_keep_prob:0.5}
            sess.run(train_step, feed_dict=train_dict)
            
        # Run loss and accuracy for training
        temp_train_loss, temp_train_acc = sess.run([loss, accuracy], feed_dict=train_dict)
        train_loss.append(temp_train_loss)
        train_accuracy.append(temp_train_acc)

        #gen test
        ###test_shuffled_ix = np.random.permutation(np.arange(len(x_train))) 
        ###x_test = x_train[test_shuffled_ix[0:10]]                
        ###y_test = y_train[test_shuffled_ix[0:10]]                        
#       x_test = x_train[shuffled_ix[0:10]]                
#        y_test = y_train[shuffled_ix[0:10]]                
#        x_test = x_test/x_zoom_factor                        
        
        # Run Eval Step
        test_dict = {x_data: x_test, y_output: y_test, dropout_keep_prob:1.0}
        temp_test_loss, temp_test_acc = sess.run([loss, accuracy], feed_dict=test_dict)
        test_loss.append(temp_test_loss)
        test_accuracy.append(temp_test_acc)
        print('Epoch: {}, Test Loss: {:.2}, Test Acc: {:.2}'.format(epoch+1, temp_test_loss, temp_test_acc))
    
    #check_shuffled_ix = np.random.permutation(np.arange(len(x_train)))
    #x_check = x_train[check_shuffled_ix[190:200]]
    #y_check = y_train[check_shuffled_ix[190:200]]
    #x_check, y_check = genCheckData()
    ###x_check, y_check = genBatchTrainData(batch_size=10, dist_start=-11,dist_end=-1,rate=1)
    x_check, y_check = genBatchTrainData(dist_start=10,dist_end=20,rate=1)
    x_check = np.array(x_check)[:batch_size]
    y_check = np.array(y_check)[:batch_size]
    #x_check = x_check/x_zoom_factor

    check_dict = {x_data: x_check, y_output: y_check, dropout_keep_prob:1.0}    
    print("y_check:", y_check)
    print(sess.run(logits_out, feed_dict=check_dict))        
    #print(sess.run(my_test_output, feed_dict=check_dict))      
    
    #Manual test
    my_test_anchor = np.array([1,9])
    x_my_check = [[my_test_anchor - ll, my_test_anchor - ul, my_test_anchor - ur, my_test_anchor - lr]]
    y_my_check = [1]
    
    my_test_anchor = np.array([8,9])
    x_my_check = np.vstack([x_my_check, [[my_test_anchor - ll, my_test_anchor - ul, my_test_anchor - ur, my_test_anchor - lr]] ])
    y_my_check.append(1)
    
    my_test_anchor = np.array([7,21])
    x_my_check = np.vstack([x_my_check, [[my_test_anchor - ll, my_test_anchor - ul, my_test_anchor - ur, my_test_anchor - lr]] ])
    y_my_check.append(0)
    check_dict = {x_data: x_my_check, y_output: y_my_check, dropout_keep_prob:1.0}    
    print("manual check:", y_my_check)
    print(sess.run(logits_out, feed_dict=check_dict))        