#The same as https://github.com/sdv-dev/CTGAN/blob/master/ctgan/data_sampler.py with a correction


import numpy as np


class DataSampler(object):
    """DataSampler samples the conditional vector and corresponding data for CTGAN."""
    
    #output_info is transformer.output_info_list
    def __init__(self, data,transactions, output_info, log_frequency):
        #self._data = data
        self._data = np.array(data, dtype='object')
        self.transactions = transactions
        self.output_info = output_info
        self.log_frequency = log_frequency
        
        #column_info is an element of transformer.output_info_list
        def is_discrete_column(column_info):
            return (len(column_info) == 1
                    and column_info[0].activation_fn == 'softmax')

        n_discrete_columns = sum(
            [1 for column_info in output_info if is_discrete_column(column_info)])

        self._discrete_column_matrix_st = np.zeros(n_discrete_columns, dtype='int32')
        self._discrete_column_matrix_indexes = np.zeros(n_discrete_columns, dtype='int32')   #added by myself

        # Store the row id for each category in each discrete column.
        # For example _rid_by_cat_cols[a][b] is a list of all rows with the
        # a-th discrete column equal value b.
        # self._rid_by_cat_cols = []

        # # Compute _rid_by_cat_cols
        # st = 0
        # for column_info in output_info:
        #     if is_discrete_column(column_info):
        #         span_info = column_info[0]
        #         ed = st + span_info.dim

        #         rid_by_cat = []
        #         for j in range(span_info.dim):
        #             rid_by_cat.append(np.nonzero(self._data[:, st + j])[0])
                   
                    
        #         self._rid_by_cat_cols.append(rid_by_cat)
        #         st = ed
        #     else:
        #         st += sum([span_info.dim for span_info in column_info])
        # assert st == data.shape[1]

        # Prepare an interval matrix for efficiently sample conditional vector
        self.max_category = max([
            column_info[0].dim
            for column_info in output_info
            if is_discrete_column(column_info)
        ], default=0)

        self._discrete_column_cond_st = np.zeros(n_discrete_columns, dtype='int32')
        self._discrete_column_n_category = np.zeros(n_discrete_columns, dtype='int32')
        self._discrete_column_category_prob = np.zeros((n_discrete_columns, self.max_category))
        self._n_discrete_columns = n_discrete_columns
        self._n_categories = sum([
            column_info[0].dim
            for column_info in output_info
            if is_discrete_column(column_info)
        ])

        st = 0
        current_id = 0
        current_cond_st = 0
        for column_info in output_info:
            if is_discrete_column(column_info):
                span_info = column_info[0]
                ed = st + span_info.dim
                # category_freq = np.sum(data[:, st:ed], axis=0)
                # if log_frequency:
                #     category_freq = np.log(category_freq + 1)
                # category_prob = category_freq / np.sum(category_freq)
                #self._discrete_column_category_prob[current_id, :span_info.dim] = category_prob
                self._discrete_column_cond_st[current_id] = current_cond_st                #offset indexes in conditional vector
                self._discrete_column_n_category[current_id] = span_info.dim               #number of categoriries in each discrete column
                self._discrete_column_matrix_indexes[current_id] = st      #added by myself
                current_cond_st += span_info.dim
                current_id += 1
                st = ed
            else:
                st += sum([span_info.dim for span_info in column_info])

    def _random_choice_prob_index(self, discrete_column_id,perm, idx):
        
        DataBatches_perm = self._data[perm]
        _discrete_column_category_prob = np.zeros((self._n_discrete_columns, self.max_category))
        st = 0
        current_id = 0
        #current_cond_st = 0
        for column_info in self.output_info:
            if len(column_info) == 1 and column_info[0].activation_fn == 'softmax':
                span_info = column_info[0]
                ed = st + span_info.dim
                category_freq = np.sum(DataBatches_perm[idx][:, st:ed], axis=0)
                if self.log_frequency:
                    category_freq = np.log(category_freq + 1)
                category_prob = category_freq / np.sum(category_freq)
                _discrete_column_category_prob[current_id, :span_info.dim] = category_prob
                current_id += 1
                st = ed
            else:
                st += sum([span_info.dim for span_info in column_info])

        probs = _discrete_column_category_prob[discrete_column_id]
        r = np.expand_dims(np.random.rand(probs.shape[0]), axis=1)
        return (probs.cumsum(axis=1) > r).argmax(axis=1)

    def sample_condvec(self, batchsize,perm, idx):
        """Generate the conditional vector for training.
        Returns:
            cond (batch x #categories):
                The conditional vector.
            mask (batch x #discrete columns):
                A one-hot vector indicating the selected discrete column.
            discrete column id (batch):
                Integer representation of mask.
            category_id_in_col (batch):
                Selected category in the selected discrete column.
        """
        DataBatches_perm = self._data[perm]
        if self._n_discrete_columns == 0:
            return None

        discrete_column_id = np.random.choice(
            np.arange(self._n_discrete_columns), batchsize)

        cond = np.zeros((batchsize, self._n_categories), dtype='float32')
        mask = np.zeros((batchsize, self._n_discrete_columns), dtype='float32')
        mask[np.arange(batchsize), discrete_column_id] = 1
        category_id_in_col = []
        for i in range(batchsize):
            col_idx = discrete_column_id[i]
            row_idx = i
            matrix_st = self._discrete_column_matrix_indexes[col_idx] 
            matrix_ed = matrix_st + self._discrete_column_n_category[col_idx]
            pick = np.argmax(DataBatches_perm[idx][row_idx, matrix_st:matrix_ed])
            category_id_in_col.append(pick)
            cond[i, pick + self._discrete_column_cond_st[col_idx]] = 1

        category_id_in_col = np.array(category_id_in_col)

        return cond, mask, discrete_column_id, category_id_in_col

    def sample_condvec2(self, batchsize,perm, idx):
        """Generate the conditional vector for training.
        Returns:
            cond (batch x #categories):
                The conditional vector.
            mask (batch x #discrete columns):
                A one-hot vector indicating the selected discrete column.
            discrete column id (batch):
                Integer representation of mask.
            category_id_in_col (batch):
                Selected category in the selected discrete column.
        """
        DataBatches_perm = self._data[perm]
        perm_g = np.arange(len(DataBatches_perm[idx]))
        np.random.shuffle(perm_g)
        DataBatches_perm_g = DataBatches_perm[idx][perm_g]
        if self._n_discrete_columns == 0:
            return None

        discrete_column_id = np.random.choice(
            np.arange(self._n_discrete_columns), batchsize)

        cond = np.zeros((batchsize, self._n_categories), dtype='float32')
        mask = np.zeros((batchsize, self._n_discrete_columns), dtype='float32')
        mask[np.arange(batchsize), discrete_column_id] = 1
        category_id_in_col = []
        for i in range(batchsize):
            col_idx = discrete_column_id[i]
            row_idx = i
            matrix_st = self._discrete_column_matrix_indexes[col_idx] 
            matrix_ed = matrix_st + self._discrete_column_n_category[col_idx]
            pick = np.argmax(DataBatches_perm_g[row_idx, matrix_st:matrix_ed])
            category_id_in_col.append(pick)
            cond[i, pick + self._discrete_column_cond_st[col_idx]] = 1

        category_id_in_col = np.array(category_id_in_col)

        return cond, mask, discrete_column_id, category_id_in_col

    def dim_cond_vec(self):
        """Return the total number of categories."""
        return self._n_categories

    def sample_data(self, perm, idx):
        """Sample data related to unique account id from original training data 
        """
        DataBatches_perm = self._data[perm]
        return DataBatches_perm[idx]

    def sample_original_condvec(self, batch):
        """Generate the conditional vector for generation use original frequency."""
        if self._n_discrete_columns == 0:
            return None

        cond = np.zeros((batch, self._n_categories), dtype='float32')
        


        for i in range(batch):
            row_idx = np.random.randint(0, len(self.transactions))
            col_idx = np.random.randint(0, self._n_discrete_columns)
            matrix_st = self._discrete_column_matrix_indexes[col_idx]             #corrected by myself
            matrix_ed = matrix_st + self._discrete_column_n_category[col_idx]
            pick = np.argmax(self.transactions[row_idx, matrix_st:matrix_ed])
            cond[i, pick + self._discrete_column_cond_st[col_idx]] = 1

        return cond

    def generate_cond_from_condition_column_info(self, condition_info, batch):
        """Generate the condition vector."""
        vec = np.zeros((batch, self._n_categories), dtype='float32')
        id_ = self._discrete_column_matrix_indexes[condition_info['discrete_column_id']]
        id_ += condition_info['value_id']
        vec[:, id_] = 1
        return vec