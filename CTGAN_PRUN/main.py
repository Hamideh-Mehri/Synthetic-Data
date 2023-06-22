from data_transformer import DataTransformer
from data_sampler import DataSampler
from ArchCTGAN import CTGAN_P
from SNIP_CTGAN import SNIP_ctgan
from prepare_data import preprocess_data_czech
import pandas as pd
import numpy as np


def main():
    df = pd.read_csv('../CTGAN/tr_by_acct_w_age.csv')
    print('preprocessing data....')
    raw_data = preprocess_data_czech(df)

    final_raw = raw_data[['amount', 'tcode','td']]

    print('transforming data...')

    transformer = DataTransformer()
    transformer.fit(final_raw, discrete_columns=('tcode'))
    data_t = transformer.transform(final_raw)      #matrix of transformed data
    output_info = transformer.output_info_list
    
    account_id_counts = raw_data['account_id'].value_counts().sort_index()
    trans_sizes = np.array(account_id_counts)
    assert sum(trans_sizes) == data_t.shape[0]
    transactions = np.split(data_t, np.cumsum(trans_sizes)[:-1])    


    log_frequency = True
    sampler = DataSampler(data_t,transactions,output_info, log_frequency)

    model = CTGAN_P(sampler, transformer)

    batchsize = 50
    target_sparsity = 0.1
    final_w_disc, final_mask_disc, final_w_gen, final_mask_gen = SNIP_ctgan(model, batchsize, target_sparsity)

    #build_generator
    generator = model.make_layers_gen(final_w_gen, final_mask_gen)
    #build discriminator
    discriminator = model.make_layers_disc(final_w_disc, final_mask_disc)

    from train import Train
    train = Train(transformer, sampler, generator, discriminator,  epochs=100)
    print('start training...')
    train.train(raw_data)

    data = train.synthesise_data3(800, raw_data)
    data.to_csv('synth.csv', index=False)

if __name__ == "__main__":
    main()


