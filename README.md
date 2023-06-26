# Generating Tabular Synthetic Data with CTGAN(Conditional Tabular GAN)
This repository contains the code for replicating the work presented in the paper "Modeling Tabular Data using Conditional GAN." 

Please note that while the original implementation in the paper uses PyTorch, this replication project utilizes Keras and TensorFlow for the implementation.

## Usage Example
insurance_ctgan.ipynb include the codes for how to train the model and generate synthetic data 

## SeqCTGAN
In SeqCTGAN, significant changes were made to the way conditional vectors and real
data are sampled. Instead of fixed batch sizes and unrelated data batches as in the
original CTGAN, the dataset is split into batches based on unique account IDs. The
batch sizes chosen are 80, 60, 40, and 20, allowing for flexibility in accommodating
PAC sizes of 10 and 20. To handle varying batch sizes, some account IDs are dupli-
cated to achieve suitable batch sizes for the PAC discriminator. For example, if there
are 117 transactions for a unique account ID, three rows with the same account ID
are randomly sampled and duplicated to increase the count to 120. This duplication
ensures that two batches of data can be formed, one with a size of 80 and another
with a size of 40.
Unlike the original CTGAN, where the conditional vector is sampled based on
categorical data frequency and real data is sampled based on the conditional vector, the process is reversed in SeqCTGAN. First, a batch of data (a sequence of transac-
tions) is randomly chosen from the split dataset. Then, the conditional vector (C1)
is sampled based on the selected transaction sequence. For each transaction row, a
categorical column is randomly chosen, and the column and its corresponding value
are used for conditioning.
During the discriminator training, the conditional vector (C1) is concatenated
with the embedding vector and passed as input to the generator. The generator’s
output is then concatenated with C1 and fed into the discriminator. The real data
is concatenated with the original conditional vector (C1) and also provided to the
discriminator.
When training the generator, the conditional vector (C2) is again sampled based
on the transaction sequence used for training the discriminator. C2 is then concate-
nated with a randomly generated vector and fed into the generator. The generator’s
output is concatenated with C2 and passed to the discriminator.

version1(v1): we use these three features for training:’transaction amount’, ’time delta’, and ’transaction code’.

| Model                 | Amt  | CF   | Tcode  | DoM/Cat  | Tcode 3G | Tcode, Date* |
|-----------------------|------|------|--------|----------|----------|--------------|
| `SeqCTGAN_v1.1` | 1241 | 9644 | 0.0003 | 0.1/0.14 | 0.006    | 0.27         |
| `SeqCTGAN_v1`   | 968  |11542 | 0.0005 | 0.1/0.14 | 0.005    | 0.27         |
| `Original CTGAN_v2` | 383  | 9185 | 0.004  | 0.1/0.14 | 0.18     | 0.21         |
| `Original CTGAN_v1(Pruned)` |174  |11083 | 0.0002 | 0.1/0.14 | 0.007    | 0.27         |
| `Original CTGAN_v1` | 232  |11765 | 0.0005 | 0.1/0.14 | 0.0086   | 0.27         |
| `BF(new)`              | 2448 | 1836 | 0.001  | 0.003/0.003 | 0.015    | 0.01         |
| `BF(old)`             | 1864 | 2738 | 0.003 | 0.02 | 0.042    | 0.03        |
| `DG`                   | 1939 |57800 | 0.007  | 0.090 | 0.132    | 0.660        |
| `TG`                   | 1931 | 4980 | 0.075  | 0.059 | 0.337    | 0.638        |
