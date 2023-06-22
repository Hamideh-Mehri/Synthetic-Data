import tensorflow as tf
import math


class ResidualLayer(tf.keras.layers.Layer):
    def __init__(self, weight, bias, weightmask, biasmask, weightname, biasname, W_maskname, b_maskname):
        super(ResidualLayer, self).__init__()
        
        self.w = tf.Variable(initial_value = weight, trainable = True, name = weightname)
        self.b = tf.Variable(initial_value = bias, trainable = True, name = biasname)

        self.weight_mask = tf.Variable(initial_value=weightmask, trainable=False, name = W_maskname)
        self.bias_mask = tf.Variable(initial_value=biasmask, trainable=False, name = b_maskname)
        
        
        
    def call(self, inputs):
        w = self.w * self.weight_mask
        b = self.b * self.bias_mask
        x = tf.matmul(inputs, w) + b
        x = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9)(x)
        x = tf.keras.layers.ReLU()(x)
        out = tf.concat([x, inputs],1)
        return out

class Dense(tf.keras.layers.Layer):
  def __init__(self, weight, bias, weightmask, biasmask, weightname, biasname, W_maskname, b_maskname):
     super(Dense, self).__init__()
     self.w = tf.Variable(initial_value = weight, trainable = True, name = weightname)
     self.b = tf.Variable(initial_value = bias, trainable = True, name = biasname)

     self.weight_mask = tf.Variable(initial_value=weightmask, trainable=False, name = W_maskname)
     self.bias_mask = tf.Variable(initial_value=biasmask, trainable=False, name = b_maskname)

  def call(self,inputs):
     w = self.w * self.weight_mask
     b = self.b * self.bias_mask
     return tf.matmul(inputs, self.w) + self.b

class DiscriminatorLayer(tf.keras.layers.Layer):
  def __init__(self,weight, bias, weightmask, biasmask, weightname, biasname, W_maskname, b_maskname):
     super(DiscriminatorLayer, self).__init__()
     self.w = tf.Variable(initial_value = weight, trainable = True, name = weightname)
     self.b = tf.Variable(initial_value = bias, trainable = True, name = biasname)

     self.weight_mask = tf.Variable(initial_value=weightmask, trainable=False, name = W_maskname)
     self.bias_mask = tf.Variable(initial_value=biasmask, trainable=False, name = b_maskname)
    
  def call(self, inputs):
     w = self.w * self.weight_mask
     b = self.b * self.bias_mask
     x = tf.matmul(inputs, w) + b
     out = tf.keras.layers.LeakyReLU(alpha = 0.2)(x)
     #out = tf.keras.layers.Dropout(rate = 0.5)(x)
     return out

class CTGAN_P(object):
    """Conditional Table GAN Synthesizer.
    This is the core class of the CTGAN project, where the different components
    are orchestrated together.
    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.
    Args:
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        
        pac (int):
            Number of samples to group together when applying the discriminator.
            Defaults to 10.
        
    """

    def __init__(self,sampler,transformer, embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256), 
                  pac=10):
        
        
        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim
        self.pac = pac
        self.sampler = sampler
        self.transformer = transformer


    def construct_weights_gen(self, trainable=True):
        
        weight = {}
        input_dim = self._embedding_dim + self.sampler.dim_cond_vec()
        bound = 1/math.sqrt(input_dim)
        w_init = tf.keras.initializers.RandomUniform(minval=-bound, maxval=bound)
        i = 1
        for output_dim in self._generator_dim:
            weight_key = 'Res_' + str(i) 
            weight[weight_key] = tf.Variable(w_init(shape=(input_dim, output_dim)))
            bias_key = 'Res_' + str(i) + '_bias'
            weight[bias_key] = tf.Variable(w_init(shape=(output_dim,)))
            input_dim += output_dim
            i +=1
        bound = 1/math.sqrt(input_dim)
        w_init = tf.keras.initializers.RandomUniform(minval=-bound, maxval=bound)
        weight['Dense'] = tf.Variable(w_init(shape=(input_dim, self.transformer.output_dimensions)))
        weight['Dense_bias'] = tf.Variable(w_init(shape=(self.transformer.output_dimensions,)))
        return weight

    def forward_pass_gen(self, weights, inputs):
        for i in range(len(self._generator_dim)):
            weight_key = 'Res_' + str(i+1) 
            bias_key = 'Res_' + str(i+1) + '_bias'
            x = tf.matmul(inputs, weights[weight_key]) + weights[bias_key]
            x = tf.nn.relu(x)
            out = tf.concat([x, inputs],1)
            inputs = out
        out = tf.matmul(inputs, weights['Dense']) + weights['Dense_bias']
        return out

    def make_layers_gen(self, weights, masks):
        
        inp_gen = tf.keras.layers.Input(shape=(self._embedding_dim + self.sampler.dim_cond_vec(),))
        x = inp_gen
        for i in range(len(self._generator_dim)):
            weight_key = 'Res_' + str(i+1) 
            bias_key = 'Res_' + str(i+1) + '_bias'
            x = ResidualLayer(weight=weights[weight_key], bias=weights[bias_key], 
                                weightmask=masks[weight_key], biasmask=masks[bias_key], 
                                weightname=weight_key, biasname=bias_key, 
                                W_maskname=weight_key+'_mask', b_maskname=bias_key+'_mask')(x)
        out = Dense(weight=weights['Dense'], bias=weights['Dense_bias'], 
                  weightmask=masks['Dense'], biasmask=masks['Dense_bias'], 
                  weightname='Dense', biasname='Dense_bias', 
                  W_maskname='Dense_mask', b_maskname='Dense_bias_mask')(x)
        model = tf.keras.models.Model(inp_gen, out) 
        return model

    def construct_weights_disc(self,trainable=True):
        weight = {}
        input_dim = self.transformer.output_dimensions + self.sampler.dim_cond_vec()
        pacdim = input_dim * self.pac

        bound = 1/math.sqrt(input_dim)
        w_init = tf.keras.initializers.RandomUniform(minval=-bound, maxval=bound)
        i = 1
        input_dim = pacdim
        for output_dim in self._discriminator_dim:
            weight_key = 'disc_' + str(i) 
            weight[weight_key] = tf.Variable(w_init(shape=(input_dim, output_dim)))
            bias_key = 'disc_' + str(i) + '_bias'
            weight[bias_key] = tf.Variable(w_init(shape=(output_dim,)))
            input_dim = output_dim
            i +=1
        bound = 1/math.sqrt(input_dim)
        w_init = tf.keras.initializers.RandomUniform(minval=-bound, maxval=bound)
        weight['Densedisc'] = tf.Variable(w_init(shape=(input_dim, 1)))
        weight['Densedisc_bias'] = tf.Variable(w_init(shape=(1,)))
        return weight

    def forward_pass_disc(self, weights, inputs):
        for i in range(len(self._discriminator_dim)):
            weight_key = 'disc_' + str(i+1) 
            bias_key = 'disc_' + str(i+1) + '_bias'
            inputs = tf.matmul(inputs, weights[weight_key]) + weights[bias_key]
            inputs = tf.nn.leaky_relu(inputs)
            
        out = tf.matmul(inputs, weights['Densedisc']) + weights['Densedisc_bias']
        return out

    def make_layers_disc(self, weights, masks):
        pacdim = (self.transformer.output_dimensions + self.sampler.dim_cond_vec()) *self.pac
        inp_disc = tf.keras.layers.Input(shape=(pacdim,))
        x = inp_disc
        for i in range(len(self._discriminator_dim)):
            weight_key = 'disc_' + str(i+1) 
            bias_key = 'disc_' + str(i+1) + '_bias'
            x = DiscriminatorLayer(weight=weights[weight_key], bias=weights[bias_key], 
                                weightmask=masks[weight_key], biasmask=masks[bias_key], 
                                weightname=weight_key, biasname=bias_key, 
                                W_maskname=weight_key+'_mask', b_maskname=bias_key+'_mask')(x)

        out = Dense(weight=weights['Densedisc'], bias=weights['Densedisc_bias'], 
                  weightmask=masks['Densedisc'], biasmask=masks['Densedisc_bias'], 
                  weightname='Densedisc', biasname='Densedisc_bias', 
                  W_maskname='Densedisc_mask', b_maskname='Densedisc_bias_mask')(x)
        model = tf.keras.models.Model(inp_disc, out) 
        return model

    
