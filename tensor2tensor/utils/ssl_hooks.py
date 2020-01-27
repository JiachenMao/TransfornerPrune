import tensorflow as tf
import pickle
'''
define all the hooks for structure sparsity training
(1) SSLEvalHook:
      SSLEvalHook is only used during the SSL training step with Group Lasso regularization.
      SSLEvalHook is executed during evaluation, which zerouts
      weights by small threshold to stablize the sparsity
(2) SSLFinetuneHook:
      SSLFinetuneHook is only used during the SSL finetune step to restore the accuracy.

For a complete SSL training procedure, a pretrained model should first do SSL training, then do SSL fine tuning:
pretrained model -> [SSL training] -> [SSL fine tuning] -> final model
'''


'''
decide what layer to prune by defined white list and black list
'''
def should_prune_v1(w, white_list, black_list):
  in_whitelist = not self.white_list or any(
      e in w.name for e in white_list)
  in_blacklist = any(e in w.name for e in black_list)
  if white_list and not in_whitelist:
    return False
  elif in_blacklist:
    return False
  return True


'''
decide what layer to prune by dimension:
  (1): only remain fc layer
  (2): discard embedding layer
'''
def should_prune_v2(w, dim=2):
  dims = w.get_shape().as_list()
  num_dims = len(dims)
  if num_dims == dim and 'body' in w.op.name and 'target_space_embedding' not in w.op.name and 'symbol_modality_' not in w.op.name:
    return True
  return False




def fill_sparisity_summary(sparsity_summary, sparisty, layer_name, layer_dim):
  if 'body' not in layer_name:
    sparsity_summary[layer_name] = sparisty
  else:
    sub_names = layer_name.split('/')
    assert(len(sub_names)>1)
    layer_idx = codec = sub_block_dim = sub_block_idx = None
    for n in sub_names:
      n = str(n)
      if n.startswith('layer_'):
        layer_idx = 1+int(n.split('_')[-1])
      elif n.startswith('_sub_block'):
        token = n.split('_')
        sub_block_dim = (int(token[5]), int(token[8]))
        sub_block_idx = (int(token[4]), int(token[7]))
      elif n=='encoder':
        codec = n
      elif n=='decoder':
        codec = n
    tf.logging.info("%s", layer_name)
    tf.logging.info("%s %s %s %s", layer_idx, codec, sub_block_dim, sub_block_idx)
    assert(layer_idx and codec and sub_block_dim and sub_block_idx)
    if 'q' in sub_names:
      layer_summary = sparsity_summary['multihead_attention']['q'][codec]
      if codec=='decoder' and 'encdec_attention' in sub_names:
        layer_summary = layer_summary['encdec_atten']
      elif codec=='decoder' and 'self_attention' in sub_names:
        layer_summary = layer_summary['self_atten']
    elif 'k' in sub_names:
      layer_summary = sparsity_summary['multihead_attention']['k'][codec]
      if codec=='decoder' and 'encdec_attention' in sub_names:
        layer_summary = layer_summary['encdec_atten']
      elif codec=='decoder' and 'self_attention' in sub_names:
        layer_summary = layer_summary['self_atten']
    elif 'v' in sub_names:
      layer_summary = sparsity_summary['multihead_attention']['v'][codec]
      if codec=='decoder' and 'encdec_attention' in sub_names:
        layer_summary = layer_summary['encdec_atten']
      elif codec=='decoder' and 'self_attention' in sub_names:
        layer_summary = layer_summary['self_atten']
    elif 'output_transform' in sub_names:
      layer_summary = sparsity_summary['multihead_attention']['output_transform'][codec]
      if codec=='decoder' and 'encdec_attention' in sub_names:
        layer_summary = layer_summary['encdec_atten']
      elif codec=='decoder' and 'self_attention' in sub_names:
        layer_summary = layer_summary['self_atten']
    elif 'ffn' in sub_names and 'conv1' in sub_names:
      layer_summary = sparsity_summary['ffn_1'][codec]
    elif 'ffn' in sub_names and 'conv2' in sub_names:
      layer_summary = sparsity_summary['ffn_2'][codec]
    if layer_idx not in layer_summary:
      num_sub_block = sub_block_dim[0]*sub_block_dim[1]
      layer_summary[layer_idx] = {'sub_block_dim': sub_block_dim, 
                                  'row_sparsity': [None]*num_sub_block,
                                  'col_sparsity': [None]*num_sub_block,
                                  'random_sparsity':[None]*num_sub_block,}
    layer_summary = layer_summary[layer_idx]
    sub_block_idx_1d = sub_block_idx[0]*sub_block_dim[1]+sub_block_idx[1]
    tf.logging.info("%s %s", layer_summary['sub_block_dim'], sub_block_dim)
    tf.logging.info("%s", layer_name)
    assert(layer_summary['sub_block_dim']==sub_block_dim)
    assert(layer_summary['row_sparsity'][sub_block_idx_1d]==None)
    assert(layer_summary['col_sparsity'][sub_block_idx_1d]==None)
    assert(layer_summary['random_sparsity'][sub_block_idx_1d]==None)
    layer_summary['row_sparsity'][sub_block_idx_1d] = sparisty['row']
    layer_summary['col_sparsity'][sub_block_idx_1d] = sparisty['col']
    layer_summary['random_sparsity'][sub_block_idx_1d] = sparisty['elt']
    sparsity_summary['total_weights'] += sub_block_dim[0]*sub_block_dim[1]
    sparsity_summary['total_nonzero_structured_weights'] += sub_block_dim[0]*sub_block_dim[1]*(1-sparisty['row'])*(1-sparisty['col'])






'''
print sparsity result of model
'''
def print_sparsity_info(sparsity_results, print_level=0, save_sparsity_info=False):
  sparsity_dict = {}
  for name, sparsity in sparsity_results.iteritems():
    layer_name = name[:-13]
    if layer_name in sparsity_dict:
      sparsity_dict[layer_name][name[-3:]] = sparsity
    else:
      sparsity_dict[layer_name] = {'elt': 0, 'col': 0, 'row': 0}
      sparsity_dict[layer_name][name[-3:]] = sparsity
  var_list = tf.trainable_variables()      
  sparsity_summary = {'ffn_1': 
                        {'encoder':{}, 'decoder':{}}, 
                      'ffn_2': 
                        {'encoder':{}, 'decoder':{}}, 
                      'multihead_attention':
                        {'output_transform': {'encoder':{}, 'decoder':{'self_atten':{}, 'encdec_atten':{}}}, 
                        'q':{'encoder':{}, 'decoder':{'self_atten':{}, 'encdec_atten':{}}}, 
                        'k':{'encoder':{}, 'decoder':{'self_atten':{}, 'encdec_atten':{}}}, 
                        'v':{'encoder':{}, 'decoder':{'self_atten':{}, 'encdec_atten':{}}}},
                      'total_weights': 0,
                      'total_nonzero_structured_weights': 0}
  tf.logging.info("##############################################")
  for layer_name, sparsities in sparsity_dict.iteritems():
    for v in var_list:
      if v.op.name == layer_name:
        dims = v.get_shape().as_list()
        if print_level>0:
          tf.logging.info("--- Layer: %s of Dimension: %s--", layer_name, dims)
        break
    if print_level>0:
      tf.logging.info("    Overall Random Sparsity: %0.4f %%", sparsities['elt']*100)
      tf.logging.info("        Row Sparsity: %0.4f %%", sparsities['row']*100)
      tf.logging.info("        Column Sparsity: %0.4f %%", sparsities['col']*100)
    fill_sparisity_summary(sparsity_summary, sparsities, layer_name, dims)
  tf.logging.info("%s", sparsity_summary)
  if save_sparsity_info:
    pickle.dump(sparsity_summary, open("sparsity_summary.p", "wb")) 
  tf.logging.info("##############################################")




class SSLEvalHook(tf.train.SessionRunHook):
      def __init__(self, zero_threshold):
        self.zero_threshold = zero_threshold
        self.sparsity = {}
        tf.logging.info("### transformer.py: _LoggerHook initialized")

      def begin(self):
        tf.logging.info("### transformer.py: _LoggerHook begin")
        
        # get weights
        weights = tf.trainable_variables()
        weights = [w for w in weights if should_prune_v2(w)]

        # establish a graph for zerout by small threshold and print sparsity statistics'
        for train_var in weights:
          # zerout by small threshold to stablize the sparsity
          sp_name = train_var.op.name
          # self.zero_threshold = max(self.zero_threshold, 2*config_params['weight_decay'])
          where_cond = tf.less(tf.abs(train_var), self.zero_threshold)
          train_var = tf.assign(train_var, 
            tf.where(where_cond, tf.zeros(tf.shape(train_var)), train_var))
          # statistics
          s = tf.nn.zero_fraction(train_var)
          self.sparsity[sp_name + '_sparsity_elt'] = s
          s = tf.nn.zero_fraction(tf.reduce_sum(tf.square(train_var), axis=0))
          self.sparsity[sp_name + '_sparsity_col'] = s
          s = tf.nn.zero_fraction(tf.reduce_sum(tf.square(train_var), axis=1))
          self.sparsity[sp_name + '_sparsity_row'] = s


      # print sparisty results after creating the session
      def after_create_session(self, session, coord):
        sparsity_results = session.run(self.sparsity)
        print_sparsity_info(sparsity_results)
        

      def before_run(self, run_context):
        pass

      def after_run(self, run_context, run_values):
        pass



class SSLInitialWeightHook(tf.train.SessionRunHook):
      def __init__(self, warm_start_from):
        tf.logging.info("ssl_hooks.py: Checkpoint dir: %s", warm_start_from)
        self.warm_start_from = warm_start_from
        

        # variable_map = {}
        # for var in tf.contrib.framework.get_trainable_variables():
        #   var_name = var.name.split(":")[0]
        #   if reader.has_tensor(var_name):
        #     tf.logging.info("Loading variable from checkpoint: %s", var_name)
        #     variable_map[var_name] = var
        #   else:
        #     tf.logging.info("Cannot find variable in checkpoint, skipping: %s",
        #                     var_name)
        # tf.train.init_from_checkpoint(ckpt_dir, variable_map)
        # tf.logging.info("### ssl_hooks.py: SSLInitialWeightHook initialized")

      '''
      Given the new layer name, return the corresponding old layer tensor
      '''
      def find_layer_tensor(self, new_name):
        name = new_name[new_name.find('/'):]
        name = self.old_model_name+name
        if '_sub_block' in name:
          name = name.split('/')
          for n in name:
            if '_sub_block' in n:
              pos_info_str = n
              break
          name.remove(pos_info_str)
          name = '/'.join(name)
          tensor = self.reader.get_tensor(name)
          # print (name)
          shape_tensor = self.var_to_shape_map[name]
          # print ('----', shape_tensor)
          pos_info = []
          for i in pos_info_str.split('_'):
            if unicode(i, 'utf-8').isnumeric():
              pos_info.append(int(i))
          input_idx, in_group, output_idx, out_group = pos_info
          # Deal with bias
          if len(shape_tensor)==1:
            col_left_bound = int(shape_tensor[0]*float(output_idx)/out_group)
            col_right_bound = int(shape_tensor[0]*float(output_idx+1)/out_group)
            tensor = tensor[col_left_bound:col_right_bound]
            return tensor/in_group
            
          # deal with kernel
          row_left_bound = int(shape_tensor[0]*float(input_idx)/in_group)
          row_right_bound = int(shape_tensor[0]*float(input_idx+1)/in_group)
          col_left_bound = int(shape_tensor[1]*float(output_idx)/out_group)
          col_right_bound = int(shape_tensor[1]*float(output_idx+1)/out_group)
          # print (row_left_bound, row_right_bound, col_left_bound, col_right_bound)
          tensor = tensor[row_left_bound:row_right_bound, col_left_bound:col_right_bound]
          return tensor
        else:
          tensor = self.reader.get_tensor(name)
          return self.reader.get_tensor(name)

      def weight_initialization_graph(self):
        self.set_weight_op = tf.no_op()
        weights = tf.trainable_variables()
        for w in weights:
          # print (w.op.name)
          # print (self.layer_tensor_map[w.op.name].shape)
          w = tf.assign(w, self.layer_tensor_map[w.op.name])
          self.set_weight_op = tf.group(self.set_weight_op, w)

      def begin(self):
        tf.logging.info("### ssl_hooks.py: SSLInitialWeightHook begin")
        self.layer_tensor_map = {}
        # Get the new model name(new ssl model) and old model name(warm start from)
        new_model_name = tf.contrib.framework.get_trainable_variables()[0]
        # print (new_model_name)
        new_model_name = new_model_name.name.split(":")[0]
        self.new_model_name = new_model_name[:new_model_name.find('/')]
        self.reader = tf.contrib.framework.load_checkpoint(self.warm_start_from)
        self.var_to_shape_map = self.reader.get_variable_to_shape_map()
        # print ('############# OLD MODEL NAME #############')
        # print (self.var_to_shape_map.keys())
        old_model_name = self.var_to_shape_map.keys()[0]
        old_model_name = old_model_name.split('/')
        for i in range(len(old_model_name)):
          if old_model_name[i]=='body':
            self.old_model_name = old_model_name[i-1]
            break
        # Generate the map from new layer name to its initilization tensor from old layer name
        for var in tf.contrib.framework.get_trainable_variables():
          var_name = var.name.split(":")[0]
          self.layer_tensor_map[var_name] = self.find_layer_tensor(var_name)
        # Generate the weight initialization graph
        self.weight_initialization_graph()

      
      def after_create_session(self, session, coord):
        tf.logging.info("### ssl_hooks.py: SSLInitialWeightHook after_create_session")
        session.run(self.set_weight_op)
        

      def before_run(self, run_context):
        pass

      def after_run(self, run_context, run_values):
        pass




class SSLFinetuneHook(tf.train.SessionRunHook):
      def __init__(self, zero_threshold):
        self.zero_threshold = zero_threshold
        self.save_sparsity_info = True
        self.sparsity = {}
        self.white_list = ['atten']
        self.black_list = ['bias']

      '''
      implement the pruning strategy and generate: 
        self.set_weight_op: operator that sets weights to 0 based on zero_threshold
        self.zero_mask: tensor that gets the zero_mask to zerout the future gradient update
        self.sparsity: tensor to get the model sparsity information
      '''
      def ssl_pruning(self, weights):
        self.set_weight_op = tf.no_op()
        self.zero_mask = {}
        self.sparsity = {}
        for w in weights:
          # w_shape = tf.map_fn(lambda x: (x), tf.shape(w))
          row, col = w.get_shape()
          w_name = w.op.name
          if 'body' in w_name:
            # get the where condition of structure sparsity
            abs_w = tf.abs(w)
            col_max_w = tf.reduce_max(abs_w, axis=0)
            row_max_w = tf.reduce_max(abs_w, axis=1)
            where_cond_col = tf.expand_dims(tf.less(col_max_w, self.zero_threshold), axis=0)
            where_cond_col = tf.tile(where_cond_col, [row, 1])
            where_cond_row = tf.expand_dims(tf.less(row_max_w, self.zero_threshold), axis=1)
            where_cond_row = tf.tile(where_cond_row, [1, col])
            where_cond = tf.logical_or(where_cond_col, where_cond_row)

            # sets weights to 0 based on zero_threshold
            w = tf.assign(w, 
              tf.where(where_cond, tf.zeros_like(w), w))
            self.set_weight_op = tf.group(self.set_weight_op, w)

            # gets the zero_mask to zerout the future gradient update
            mask = tf.where(where_cond, tf.zeros_like(w), tf.ones_like(w))
            self.zero_mask[w_name+'_mask'] = mask

            # get the model sparsity information
            s = tf.nn.zero_fraction(mask)
            self.sparsity[w_name + '_sparsity_elt'] = s
            s = tf.nn.zero_fraction(tf.reduce_sum(tf.square(w), axis=0))
            self.sparsity[w_name + '_sparsity_col'] = s
            s = tf.nn.zero_fraction(tf.reduce_sum(tf.square(w), axis=1))
            self.sparsity[w_name + '_sparsity_row'] = s


      # generate the operation zerout weights with masks during each training steps
      def zerout_weights_with_masks(self, weights):
        self.mask_placeholders = {}
        self.zerout_op = tf.no_op()
        for w in weights:
          # w_shape = tf.shape(w)
          w_shape = w.get_shape()
          w_name = w.op.name
          # store the mask placeholder for future assignment
          mask_placeholder = tf.placeholder(w.dtype, shape=w_shape)
          self.mask_placeholders[w_name+'_mask']=mask_placeholder
          # update weight
          updated_weight = tf.multiply(w, mask_placeholder)
          op = tf.assign(w, updated_weight)
          self.zerout_op = tf.group(self.zerout_op, op)
      

      def begin(self):
        tf.logging.info("### transformer.py: _SSLfinetuneHook begin")
        
        # get weights
        weights = tf.trainable_variables()
        weights = [w for w in weights if should_prune_v2(w)]

        # establish graphs to prune weight, get mask, and print sparsity before session run
        self.ssl_pruning(weights)
        # establish a graph to zerout weights with masks during each training steps
        self.zerout_weights_with_masks(weights)
        

      def after_create_session(self, session, coord):
        # sets weights to 0 based on zero_threshold
        session.run(self.set_weight_op)

        # get zero masks to zerout weights with masks during each training steps
        self.zero_masks_np = session.run(self.zero_mask)

        # print the final structure sparsity
        sparsity_results = session.run(self.sparsity)
        print_sparsity_info(sparsity_results, save_sparsity_info=self.save_sparsity_info)
        self.save_sparsity_info=False

      def before_run(self, run_context):
        '''
        Unlike wei's realization which apply zero masks to gradients before weights update
        I apply the zero masks on the weights after the weights are updated with gradients
        The aforementioned two realization achieves the same result
        '''
        zero_masks_dict={}
        # for i, placeholder in enumerate((self.mask_placeholders.values())):
        #   zero_masks_dict[placeholder] = self.zero_masks_np[i]
        for name, placeholder in self.mask_placeholders.iteritems():
          if name in self.zero_masks_np.keys():
            zero_masks_dict[placeholder] = self.zero_masks_np[name]
          else:
            raise ValueError('Can not found zero_mask of layer: %s', name)
        return tf.train.SessionRunArgs(self.zerout_op, feed_dict=zero_masks_dict)


      def after_run(self, run_context, run_values):
        # check the weight sparsity for debug purpose
        # tf.logging.info("### ssl_hooks.py: after_run executed tensors: ----%s", run_values.results)
        pass


































