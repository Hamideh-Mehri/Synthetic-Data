import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import random
import pandas as pd


class Train(object):
    """
    generator_lr (float):
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay (float):
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr (float):
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay (float):
            Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original CTGAN implementation.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 300."""
            
    def __init__(self, transformer, sampler, generator, discriminator, embedding_dim=128, batch_size=500, generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6,discriminator_steps=3,
                 log_frequency=True, verbose=True, epochs=100, pac=10):
        assert batch_size % 2 == 0
        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self._embedding_dim = embedding_dim
        self._transformer = transformer
        self._sampler = sampler
        self.pac = pac
        self.generator = generator
        self.discriminator = discriminator

    
    
    def _gumbel_softmax(self, logits, tau=1.0, hard=False, dim=-1):
        """Samples from the Gumbel-Softmax distribution
        :cite:`maddison2016concrete`, :cite:`jang2016categorical` and
        optionally discretizes.
        Parameters
        ----------
        logits: tf.Tensor
            Un-normalized log probabilities.
        tau: float, default=1.0
            Non-negative scalar temperature.
        hard: bool, default=False
            If ``True``, the returned samples will be discretized as
            one-hot vectors, but will be differentiated as soft samples.
        dim: int, default=1
            The dimension along which softmax will be computed.
        Returns
        -------
        tf.Tensor
            Sampled tensor of same shape as ``logits`` from the
            Gumbel-Softmax distribution. If ``hard=True``, the returned samples
            will be one-hot, otherwise they will be probability distributions
            that sum to 1 across ``dim``.
        """

        gumbel_dist = tfp.distributions.Gumbel(loc=0, scale=1)
        gumbels = gumbel_dist.sample(tf.shape(logits))
        gumbels = (logits + gumbels) / tau
        output = tf.nn.softmax(gumbels, dim)

        if hard:
            index = tf.math.reduce_max(output, 1, keepdims=True)
            output_hard = tf.cast(tf.equal(output, index), output.dtype)
            output = tf.stop_gradient(output_hard - output) + output
        return output
        
    
    
    
    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(tf.math.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

        return tf.concat(data_t, axis=1)

    def cross_entropy_loss(self, data, c, m, output_info):
        """Compute the cross entropy loss on the fixed discrete column."""
        loss = []
        st = 0
        st_c = 0
        for column_info in output_info:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != 'softmax':
                    # not discrete column
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    #c is the conditional vector
                    labels=c[:, st_c:ed_c]
                    #data is generated by generator
                    logits=data[:, st:ed]
                    tmp = tf.nn.softmax_cross_entropy_with_logits(
                        labels,
                        logits)
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = tf.stack(loss, axis=1)
        #we are interested in the loss for the feature that is conditioned on 
        m1 = tf.cast(m, dtype=tf.float32)
        return tf.reduce_mean(loss * m1)

    def calc_gradient_penalty(self, real_cat, fake_cat, gp_lambda=10):
        #random alpha(between 0 and 1) for each input batch to discriminator
        alpha = tf.random.uniform([real_cat.shape[0] // self.pac, 1, 1], 0., 1.)
        alpha = tf.tile(alpha, tf.constant([1, self.pac, real_cat.shape[1]], tf.int32))
        alpha = tf.reshape(alpha, [-1, real_cat.shape[1]])
        
        interpolates = alpha * real_cat + ((1 - alpha) * fake_cat)
        pacdim = self.pac * real_cat.shape[1]
        interpolates_disc = tf.reshape(interpolates,[-1, pacdim])

        with tf.GradientTape() as tape:
            tape.watch(interpolates_disc)
            pred_interpolates = self.discriminator(interpolates_disc)

        gradients = tape.gradient(pred_interpolates, interpolates_disc)

        gradients_view = tf.norm(tf.reshape(gradients, [-1, self.pac * real_cat.shape[1]]), axis=1) - 1
        gradient_penalty = tf.reduce_mean(tf.square(gradients_view)) * gp_lambda
        return gradient_penalty
    
    def train(self, raw_data):
        
        optimizerG = tf.keras.optimizers.Adam(learning_rate = self._generator_lr, beta_1=0.5, beta_2=0.9, decay = self._generator_decay)
        optimizerD = tf.keras.optimizers.Adam(learning_rate = self._discriminator_lr, beta_1=0.5, beta_2=0.9, decay = self._discriminator_decay)

        mean = tf.zeros(shape=(self._batch_size, self._embedding_dim), dtype=tf.float32)
        std = mean + 1
        steps_per_epoch = max(len(raw_data)// self._batch_size, 1)

        data_dim = self._transformer.output_dimensions
        dim_cond_vec = self._sampler.dim_cond_vec()
        #dim of the input to discriminator
        input_dim = data_dim + dim_cond_vec
        pacdim = input_dim * self.pac

        for i in range(self._epochs):
            for id_ in range(steps_per_epoch):
                for n in range(self._discriminator_steps):
                    fakez = tf.random.normal(shape=(self._batch_size, self._embedding_dim), mean=mean, stddev=std)
                    condvec = self._sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = self._sampler.sample_data(self._batch_size, col, opt)
                    else:
                        c1, m1, col, opt = condvec
                        c1 = tf.convert_to_tensor(np.array(c1))
                        c1 = tf.cast(c1, dtype=tf.float32)
                        #c1 = tf.identity(c1, name=None) # Optional, just to ensure a new tensor is created

                        m1 = tf.convert_to_tensor(np.array(m1))
                        m1 = tf.cast(m1, dtype=tf.int32)
                        #m1 = tf.identity(m1, name=None) # Optional, just to ensure a new tensor is created
                        fakez = tf.concat([fakez, c1], axis=1)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._sampler.sample_data(self._batch_size, col[perm], opt[perm])
                        c2 = tf.gather(c1, indices=perm)
                        
                    fake = self.generator(fakez)
                    fakeact = self._apply_activate(fake)

                    if c1 is not None:
                       fake_cat = tf.concat([fakeact, c1], axis=1)
                       real_cat = tf.concat([real, c2], axis=1)
                    else:
                       fake_cat = fakeact
                       real_cat = real

                    # reshape the data for packed discriminator
                    fake_cat_disc = tf.reshape(fake_cat,[-1, pacdim])
                    real_cat_disc = tf.reshape(real_cat, [-1, pacdim])

                    with tf.GradientTape() as tape:
                        y_fake = self.discriminator(fake_cat_disc, training=True)
                        y_real = self.discriminator(real_cat_disc, training=True)
                        pen = self.calc_gradient_penalty(real_cat, fake_cat, 10)
                        loss_d = -(tf.reduce_mean(y_real) - tf.reduce_mean(y_fake)) + pen
                    grads_disc = tape.gradient(loss_d, self.discriminator.trainable_variables)
                    optimizerD.apply_gradients(zip(grads_disc, self.discriminator.trainable_variables))

                fakez = tf.random.normal(shape=(self._batch_size, self._embedding_dim), mean=mean, stddev=std)
                condvec = self._sampler.sample_condvec(self._batch_size)
                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                    
                else:
                    c1, m1, col, opt = condvec
                    c1 = tf.convert_to_tensor(np.array(c1))
                    c1 = tf.cast(c1, dtype=tf.float32)

                    m1 = tf.convert_to_tensor(np.array(m1))
                    m1 = tf.cast(m1, dtype=tf.int32)
                
                    fakez = tf.concat([fakez, c1], axis=1)

                with tf.GradientTape() as tape:
                    fake = self.generator(fakez)
                    fakeact = self._apply_activate(fake)

                    if c1 is not None:
                       fake_temp = tf.concat([fakeact, c1], axis=1)
                       y_fake = self.discriminator(tf.reshape(fake_temp,[-1, pacdim]))
                    else: 
                       y_fake = self.discriminator(tf.reshape(fakeact,[-1, pacdim]))

                    if condvec is None:
                       cross_entropy = 0
                    else:
                       output_info = self._transformer.output_info_list
                       cross_entropy = self.cross_entropy_loss(fake, c1, m1, output_info)

                    #loss_g = -tf.reduce_mean(y_fake) + cross_entropy
                    loss_g = -tf.reduce_mean(y_fake) + cross_entropy
                grads_gen = tape.gradient(loss_g, self.generator.trainable_variables)
                optimizerG.apply_gradients(zip(grads_gen, self.generator.trainable_variables))

            print(f'Epoch {i+1}, Loss G: {loss_g.numpy(): .4f}, Loss D: {loss_d.numpy(): .4f}', flush=True)

    def synthesise_data(self, n, condition_column=None, condition_value=None):
        """Sample data similar to the training data.
        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.
        Args:
            n (int):
                Number of rows to sample.
            condition_column (string):
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.
        Returns:
                pandas.DataFrame
        """
        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(condition_column, condition_value)
            global_condition_vec = self._sampler.generate_cond_from_condition_column_info(condition_info, self._batch_size)
        else:
            global_condition_vec = None
        
        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            mean = tf.zeros(shape=(self._batch_size, self._embedding_dim), dtype=tf.float32)
            std = mean + 1
            fakez = tf.random.normal(shape=(self._batch_size, self._embedding_dim), mean=mean, stddev=std)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self._sampler.sample_original_condvec(self._batch_size)

            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = tf.convert_to_tensor(np.array(c1))
                c1 = tf.cast(c1, dtype=tf.float32)
                fakez = tf.concat([fakez, c1], axis=1)
            fake = self.generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]
        return self._transformer.inverse_transform(data)

    def synthesise_data3(self,n, raw_data, condition_column=None, condition_value=None):
        """Sample data similar to the training data.
        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.
        Args:
            n (int):
                Number of transaction sequences.
            condition_column (string):
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.
        Returns:
                pandas.DataFrame
        """
        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(condition_column, condition_value)
            global_condition_vec = self._sampler.generate_cond_from_condition_column_info(condition_info, length_of_data)
        else:
            global_condition_vec = None
        
        unique_account_ids = raw_data['account_id'].unique()
        date_time = []
        trans_length = []
        data = []
        for i in range(n):
            idx = random.randint(0, len(self._sampler.transactions))
            account_id = unique_account_ids[idx]
            trans_length.append(len(self._sampler.transactions[idx]))
            length_of_seq = self._sampler.transactions[idx].shape[0]
            random_row = raw_data.sample(n=1)
            first_row_datetime = random_row['datetime'].values[0]
            #first_row_datetime = raw_data[raw_data['account_id'] == account_id].iloc[0]['datetime']
            
            date_time.append(first_row_datetime)
            mean = tf.zeros(shape=(length_of_seq, self._embedding_dim), dtype=tf.float32)
            std = mean + 1
            fakez = tf.random.normal(shape=(length_of_seq, self._embedding_dim), mean=mean, stddev=std)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self._sampler.sample_original_condvec3(idx)

            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = tf.convert_to_tensor(np.array(c1))
                c1 = tf.cast(c1, dtype=tf.float32)
                fakez = tf.concat([fakez, c1], axis=1)
            fake = self.generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.numpy())

        data = np.concatenate(data, axis=0)
        #data = data[:n]
        synth = self._transformer.inverse_transform(data)
        #set account_id's
        count = 0
        curr_index = 0
        account_ids = []

        for _ in range(len(synth)):
            count += 1
            if count > trans_length[curr_index]:
                curr_index += 1
                count = 1
            account_ids.append(curr_index + 1)

        synth['account_id'] = account_ids
        synth['td'] = synth['td'].apply(lambda x: 0 if x < 0 else round(x))
        synth['cumulative_td'] = synth.groupby('account_id')['td'].cumsum()
        # Initialize an empty list to store the final 'datetime' values
        datetime_values = []
       
        # Iterate over the rows of the sorted DataFrame
        for index, row in synth.iterrows():
            account_id = row['account_id']
            cumulative_td = int(row['cumulative_td'])
            first_datetime = date_time[account_id - 1]  # Subtract 1 to account for 0-based indexing
            
            # Calculate the datetime value based on the first row datetime and cumulative td
            datetime_value = first_datetime + pd.Timedelta(cumulative_td, unit='D')  # Assuming 'td' is in days
            datetime_values.append(datetime_value)

        # Assign the datetime values to the new 'datetime' column in the DataFrame
        synth['datetime'] = datetime_values
        return synth
                
            

