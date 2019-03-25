# when using tensorflow
# ------------------checkpoints---------------------------------
saver = tf.train.Saver(tf.global_variables(),max_to_keep=1)   # save variables before tf.session

checkpoint_path = os.path.join(Path, 'model.ckpt') # during train
saver.save(session, checkpoint_path, global_step=step)  #step: the number of training, the i th iteration

saver = tf.train.Saver(tf.global_variables()) # recover all veriables
moudke_file=tf.train.latest_checkpoint('PATH')
saver.restore(sess,moudke_file)

def read_checkpoint(): # load part of checkpoints
    w = []
    checkpoint_path = '.........../model.ckpt-17000'
    reader = tf.train.NewCheckpointReader(checkpoint_path)
    var = reader.get_variable_to_shape_map()
    for key in var:
        if 'weights' in key and 'conv' in key and 'Mo' not in key:
            print('tensorname:', key)
    #     # print(reader.get_tensor(key))

var=[v for v in weight_pruned  if v.op.name=='WRN/conv1/weights']    # assign part of the value to network
conv1_temp=tf.convert_to_tensor(conv1,dtype=tf.float32) # data structure <name,shape,dtype>
sess.run(tf.assign(var[0],conv1_temp)) # store all value of new network in weight_pruned


# ------------------weights kernal------------------------------
