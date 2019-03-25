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
# mwthod from https://github.com/cnnpruning/Kernel-Pruning/blob/master/kernel_prune.py
total = 0
total_kernel=0
for m in model.modules(): # count total weights for a kernal
    if isinstance(m, nn.Conv2d):
        total += m.weight.data.numel()
        oc,ic,h,w=m.weight.size()
        total_kernel+=m.weight.data.numel()/(w*h)
conv_weights = torch.zeros(total).cuda()
conv_max_weights = torch.zeros(total).cuda()

index = 0
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        size = m.weight.data.numel()
        conv_weights[index:(index+size)] = m.weight.data.view(-1).abs().clone()
        oc,ic,h,w=m.weight.size()
        weight_max=torch.max(m.weight.data.abs().view(oc,ic,w*h),-1)[0].view(oc,ic,1,1).expand(oc,ic,h,w)
        conv_max_weights[index:(index+size)] = weight_max.contiguous().view(-1).clone()
        index += size

y, i = torch.sort(conv_max_weights)
thre_index = int(total * args.percent)
# --------------- prune --------------------
thre = y[thre_index]
zero_flag=False
pruned = 0
print('Percent {} ,Pruning threshold: {}'.format(args.percent,thre))
index = 0
for k, m in enumerate(model.modules()):
    if isinstance(m, nn.Conv2d):
        size = m.weight.data.numel()
        oc,ic,h,w=m.weight.size()
        mask = conv_max_weights[index:(index+size)].gt(thre).float().cuda().detach().view(oc,ic,h,w)

        # weight_copy = m.weight.data.abs().clone()
        # mask = weight_copy.gt(thre).float().cuda()

        pruned = pruned + mask.numel() - torch.sum(mask)
        m.weight.data.mul_(mask)
        index += size
        if int(torch.sum(mask)) == 0:
            zero_flag = True
        print('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.
              format(k, mask.numel(), int(torch.sum(mask))))
print('Total conv params: {}, Pruned conv params: {}, Pruned ratio: {}'.format(total, pruned, pruned / total))
