import pandas as pd
from tqdm import tqdm
import numpy as np

K = 7
len_kmer = 16491

chip_seq_file = 'ChIP_with_Sequence.xlsx'
data = pd.read_excel(chip_seq_file)
sequence_list = data['sequence'].values

num_kmer = 0
kmer_dict = {}

for seq in tqdm(sequence_list):
    for index_k in range(len(seq) - K + 1):
        temp_seq = seq[index_k:index_k + K]
        if temp_seq not in kmer_dict:
            kmer_dict[temp_seq] = num_kmer
            num_kmer += 1

print(num_kmer)
np.save('kmer_dict.npy', kmer_dict)

data = data[['seqnames', 'scaled_correlation_of_NNv1_Age', 'sequence']]
data.to_csv('data.csv')


