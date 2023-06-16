from ctgan import CTGAN
from data_transformer import DataTransformer
from data_sampler import DataSampler
from train import Train
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

from prepare_data import preprocess_data_czech

def divide_into_multiples(number):
    multiples_80 = number // 80
    remainder_80 = number % 80

    multiples_60 = remainder_80 // 60
    remainder_60 = remainder_80 % 60

    multiples_40 = remainder_60 // 40
    remainder_40 = remainder_60 % 40

    multiples_20 = remainder_40 // 20
    remainder_20 = remainder_40 % 20
    result = multiples_80 * 80 + multiples_40 * 40 + multiples_60 * 60 + multiples_20 * 20
    if remainder_20 > 0:
        result += remainder_20
    l = [multiples_80, multiples_60, multiples_40, multiples_20, remainder_20]
    return l

def construct_list(numbers_list):
    result = []
    for idx, num in enumerate(numbers_list):
        if idx == 0:
            result.extend([80] * num)
        elif idx == 1:
            result.extend([60] * num)
        elif idx == 2:
            result.extend([40] * num)
        elif idx == 3:
            result.extend([20] * (num))
    return result

def main():
    print('preprocess data')
    df_raw = pd.read_csv('../CTGAN/tr_by_acct_w_age.csv')
    raw_data = preprocess_data_czech(df_raw)
    df = raw_data[['account_id', 'type', 'amount', 'tcode', 'month', 'dow','year', 'dtme_cat', 'age_group', 'td']]

    result_df = pd.DataFrame()
    Batch_sizes = []                                #input to __init__ function from Train class as batch_size
    unique_account_ids = df['account_id'].unique()
    print('construct batches')
    for account_id in unique_account_ids:
        account_rows = df[df['account_id'] == account_id]
        # Check the number of rows for the account ID
        num_rows = len(account_rows)
        l = divide_into_multiples(num_rows)
        num_duplicates = 20 - l[-1]
        duplicated_rows = account_rows.sample(n=num_duplicates, replace=True)
        account_rows = pd.concat([account_rows, duplicated_rows], ignore_index=True)
        num_rows_new = len(account_rows)
        l_new = divide_into_multiples(num_rows_new)
        result_df = pd.concat([result_df, account_rows], ignore_index=True)
        Batch_sizes.extend(construct_list(l_new))

    final_raw = raw_data[['amount', 'tcode','month', 'dow', 'year','dtme_cat', 'age_group', 'td']]
    final_raw_dup = result_df[['amount','tcode', 'month', 'dow', 'year','dtme_cat', 'age_group', 'td']]
    transformer = DataTransformer()
    #we can fit the transformer with the original data and then transform the duplicated data
    transformer.fit(final_raw, discrete_columns=('tcode','month', 'dow','year', 'dtme_cat', 'age_group'))
    data_t = transformer.transform(final_raw_dup)      #matrix of transformed data
    trans_t = transformer.transform(final_raw) 
    output_info = transformer.output_info_list

    # Load an object
    # with open('object.pkl', 'rb') as file:
    #     transformer = pickle.load(file)

    # # Load a matrix
    # with open('matrix.pkl', 'rb') as file:
    #     data_t = pickle.load(file)

    # with open('matrix2.pkl', 'rb') as file:
    #     trans_t = pickle.load(file)

    output_info = transformer.output_info_list

    # Validate that the sum of split_sizes is equal to the total number of rows
    assert sum(Batch_sizes) == data_t.shape[0]

    # Split the data_array into arrays of the specified sizes
    DataBatches = np.split(data_t, np.cumsum(Batch_sizes)[:-1])    #DataBatches is given to datasampler

    # account_id_counts = raw_data['account_id'].value_counts().sort_index()
    # trans_sizes = np.array(account_id_counts)
    # assert sum(trans_sizes) == trans_t.shape[0]
    # transactions = np.split(trans_t, np.cumsum(trans_sizes)[:-1])
    transactions = trans_t

    log_frequency = True
    sampler = DataSampler(DataBatches,transactions, output_info, log_frequency)

    model = CTGAN()
    generator  = model.make_generator(sampler, transformer)
    discriminator = model.make_discriminator(sampler, transformer)

    train = Train(transformer, sampler, generator, discriminator,Batch_sizes ,pac=10, epochs= 5)
    train.train()

    data = train.synthesise_data(4000, 80)
    data.to_csv('synth.csv', index=False)

if __name__ == "__main__":
    main()


