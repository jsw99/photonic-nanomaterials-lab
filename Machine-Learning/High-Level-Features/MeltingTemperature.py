from Bio.SeqUtils import MeltingTemp as mt
from Bio.Seq import Seq
import numpy as np
import pandas as pd

seq_string_list = ['TATTATTATTAT', 'TTATTTTTTATT', 'ATTTTTTTTTTA', 'TTATATTATATT', 'CCACCACCACCA',
            'GCGCGCGCGCGC', 'TGTGTTTTGTGT', 'TTTTCCCCCCCC', 'TTTCCCCCCCCC', 'CCTCCCCCCCCT',
            'GTGTGTGTGTGT', 'ATTTATTTATTT', 'CCCCCTCCCCCT', 'TTTCCCCCCCCT', 'CCTTTCCCCCCT',
            'GGGGTTTTGGGG', 'AACACCACACAC', 'TTTTCCCCCCCT', 'ACACACACAACG', 'AACACACCACAG',
            'ACACACCATCAG', 'AGCAGCACACGA', 'ACGCCAACACAT', 'AACACACACAGC', 'AGCACCAGACAG',
            'ACGCACCGACAG']

seq_list = []
mt_wallace_list = [] # Rule of thumb
mt_GC_list = [] # Empirical formulas based on GC content
mt_breslauer_list = [] # Breslauer '86
mt_sugimoto_list = [] # Sugimoto '96

for seq_string in seq_string_list:
    seq = Seq(seq_string)
    seq_list.append(seq)
    mt_wallace_list.append(np.round(mt.Tm_Wallace(seq),2))
    mt_GC_list .append(np.round(mt.Tm_GC(seq), 2))
    mt_breslauer_list.append(np.round(mt.Tm_NN(seq, nn_table=mt.DNA_NN1), 2))
    mt_sugimoto_list.append(np.round(mt.Tm_NN(seq, nn_table=mt.DNA_NN2), 2))

melting_temp = {'Sequence': seq_string_list,
                'Tm_Wallace': mt_wallace_list,
                'Tm_GC': mt_GC_list,
                'Tm_Breslauer': mt_breslauer_list,
                'Tm_Sugimoto': mt_sugimoto_list}

df = pd.DataFrame(melting_temp)
print(df)

#export DataFrame to text file
with open('MeltingTemperature.txt', 'a') as f:
    df_string = df.to_string(header=True, index=False)
    f.write(df_string)