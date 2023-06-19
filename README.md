# Generating Tabular Synthetic Data with CTGAN(Conditional Tabular GAN)
This repository contains the code for replicating the work presented in the paper "Modeling Tabular Data using Conditional GAN." 

Please note that while the original implementation in the paper uses PyTorch, this replication project utilizes Keras and TensorFlow for the implementation.

# Usage Example
insurance_ctgan.ipynb include the codes for how to train the model and generate synthetic data 

# SeqCTGAN
In the proposed data generation approach, the dataset is split into batches, with each batch containing a sequence of transactions for a unique account ID. The batch sizes chosen are 80, 60, 40, and 20, allowing for flexibility in accommodating PAC sizes of 10 and 20.

To handle varying batch sizes, some account IDs are duplicated to achieve batch sizes suitable for the PAC discriminator. For example, if there are 117 transactions for a unique account ID, three rows are randomly sampled from the transactions with the same account ID to increase the count to 120. This duplication ensures that there can be two batches of data, one with a size of 80 and another with a size of 40.

Rather than sampling the conditional vector based on the frequency of categorical data and then sampling real data based on the conditional vector (as done in the original CTGAN), the process is reversed. First, a batch of data(a transaction sequence) is chosen randomly from the split dataset. Then, the conditional vector (C1) is sampled based on the sampled transaction sequence(for each row of transaction, a categorical column is chosen randomly, and the column and the value in that column is chosen for conditioning).

During the training of the discriminator, the conditional vector is shuffled(C2) and concatenated with the embedding vector. The resulting vector is fed into the generator. The output of the generator is concatenated with the shuffled conditional vector(C2) and fed into the discriminator. The real data is concatenated with the original conditional vector(C1) and also fed into the discriminator.

When training the generator, the conditional vector is sampled based on a shuffle of the transaction sequence(C3). Then, the conditional vector is shuffled again(C4) and concatenated with a random vector, which is fed into the generator. The output of the generator is concatenated with the shuffled conditional vector(C4) and fed into the discriminator.

For generating synthetic data, a row of the whole dataset is chosen randomly, and a categorical column is chosen randomly, and the conditional vector is constructed based on the value of the categorical column on the chosen row. 

## synth1.csv
Column of dataset used for training: 'amount', 'tcode', 'month', 'dow', 'year', 'dtme_cat', 'age_group', 'td'
procedure of training: the same as explained above
## synth2.csv
Column of dataset used for training: 'amount', 'tcode','month', 'dow', 'year','day'

During the training of the discriminator,the original conditional vector(C1) is concatenated with the embedding vector. The resulting vector is fed into the generator. The output of the generator is concatenated with C1 and fed into the discriminator. The real data is concatenated with the original conditional vector(C1) and also fed into the discriminator.

When training the generator, again the conditional vector is sampled based on the transaction sequence(C3). Then, the conditional vector is shuffled (C4) and concatenated with a random vector, which is fed into the generator. The output of the generator is concatenated with the shuffled conditional vector(C4) and fed into the discriminator.

For generating synthetic data, an account_id is chosen randomly, for each row of that transaction, a categorical column is sampled randomly, and the conditional vector is constructed based on the categorical column and the value of the categorical column on that row. 

