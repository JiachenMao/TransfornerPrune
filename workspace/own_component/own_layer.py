import tensorflow as tf

'''
Decouple a dense layer into several block.
Params:
  input_tensor: input tensor
  out_size: output tensor dimension
  in_group: number of sub-blocks the original weight matrix is divided to from the input dimension
  out_group: number of sub-blocks the original weight matrix is divided to from the output dimension
  name: name of the original dense layer
  use_bias: whether to use bias or not
  split_dim: the dimension of the input and output to be decoupled (default as 1)
  group_dim: the dimension of group to be added after each blocked output is calculated (default as 0)
Return:
  output_tensor: output tensor
'''
def dense_block_wise_decouple(input_tensor, out_size, in_group, out_group, name, use_bias, split_dim=1, group_dim=0):
  # print ('### dense_block_wise_decouple: dimension check %s' %(input_tensor.shape))
  assert(out_size%out_group == 0)
  block_wise_outputs = [[] for _ in range(out_group)]
  split_inputs = tf.split(value=input_tensor, num_or_size_splits=in_group, axis=split_dim) # split_dim stands for input dimension
  for input_idx in range(len(split_inputs)):
    for output_idx in range(out_group):
      # with tf.variable_scope("%s/sub_block_in_%d_out_%d" % (name, input_idx, output_idx)):
      block_wise_output = tf.layers.dense(split_inputs[input_idx], out_size/out_group, use_bias=use_bias, name="%s/_sub_block_in_%d_%d_out_%d_%d_" % (name, input_idx, in_group, output_idx, out_group))
      block_wise_outputs[output_idx].append(block_wise_output)
  # Ruduce sum for all the sub-block output with the same position
  for output_idx in range(out_group):
    block_wise_outputs[output_idx] = tf.reduce_sum(block_wise_outputs[output_idx], axis=group_dim) # group_dim stands for each group
  # Concatenate all the block-wise outputs
  output_tensor = tf.concat(block_wise_outputs, axis=split_dim) # split_dim stands for output dimension
  # if use_bias:
  #   bias = tf.random.normal(shape=output_tensor.shape)
  #   output_tensor = tf.nn.bias_add(output_tensor, bias, name=name+'_bias')
  return output_tensor