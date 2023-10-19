from myctgan import CTGAN
from data_transformer import DataTransformer
from data_sampler import DataSampler
from train import Train
import numpy as np
import tensorflow as tf
import pandas as pd
import json
import pickle

from prepare_data import preprocess_data_czech



def main():
    with tf.device('/gpu:0'):
        df = pd.read_csv('../DATA/tr_by_acct_w_age.csv')
        synth_bf = pd.read_csv('../DATA/banksformer.csv')
        grouped = synth_bf.groupby('account_id')
        print('preprocessing data....')
        raw_data = preprocess_data_czech(df)
        raw = raw_data[['account_id', 'type', 'amount', 'tcode', 'datetime', 'year', 'month', 'dow', 'day', 'dtme','td']]
        df = raw.copy()
        # Remove the first part ('debit__' or 'credit__') from the 'tcode' column using .loc
        #df.loc[:,'tcode'] = df['tcode'].str.replace('^(DEBIT__|CREDIT__)', '', regex=True)
        final_raw = df[['amount', 'tcode', 'td', 'day', 'dow', 'dtme', 'month']]
        print('transforming data...')
        transformer = DataTransformer()
        
        transformer.fit(final_raw, discrete_columns=('tcode','day','dow','dtme','month'))     
        data_t, mean_dict, std_dict, df = transformer.transform(final_raw)                #matrix of transformed data
        output_info = transformer.output_info_list

        account_id_counts = raw_data['account_id'].value_counts().sort_index()
        trans_sizes = np.array(account_id_counts)
        assert sum(trans_sizes) == data_t.shape[0]
        transactions = np.split(data_t, np.cumsum(trans_sizes)[:-1])    


        log_frequency = True
        sampler = DataSampler(data_t,transactions,output_info, log_frequency)

        model = CTGAN()
        generator  = model.make_generator(sampler, transformer)
        discriminator = model.make_discriminator(sampler, transformer)

        train = Train(transformer, sampler, generator, discriminator, epochs=100)
        experiment_name = 'exp_B15'
        train.train(raw_data, 'exp_B15')
        print('synthesize data')
        #data, visdata = train.synthesise_data_bank3(2000, raw_data, mean_dict, std_dict, df)
        data, visdata = train.synthesise_data_bank_externaltcode_v3(2300, raw_data, mean_dict, std_dict, df, grouped)
        print('finish')
        filename = '../DATA/synth_' + experiment_name +'.csv'
        data.to_csv(filename, index=False)
        filename_vis = '../DATA/'+ experiment_name + '_vis' + '.pickle'
        with open(filename_vis, 'wb') as f:
              pickle.dump(visdata, f)

if __name__ == "__main__":
    main()


    
