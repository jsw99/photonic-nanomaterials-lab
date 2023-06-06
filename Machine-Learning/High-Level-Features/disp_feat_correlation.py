from Bio.SeqUtils import MeltingTemp as mt
from Bio.SeqUtils import molecular_weight
from Bio.Seq import Seq
from seqfold import dg
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

seq_string_list = [
            'TATTATTATTAT', 'TTATTTTTTATT', 'ATTTTTTTTTTA', 'TTATATTATATT', 'CCACCACCACCA',
            'GCGCGCGCGCGC', 'TGTGTTTTGTGT', 'TTTTCCCCCCCC', 'TTTCCCCCCCCC', 'CCTCCCCCCCCT',
            'GTGTGTGTGTGT', 'ATTTATTTATTT', 'CCCCCTCCCCCT', 'CCTTTCCCCCCT', 'GGGGTTTTGGGG',
            'AACACCACACAC', 'ACACACACAACG', 'AACACACCACAG', 'ACACACCATCAG', 'AGCAGCACACGA',
            'ACGCCAACACAT', 'AACACACACAGC', 'AGCACCAGACAG', 'ACGCACCGACAG',
            'TTTTCCCCCCCT', 'TTTCCCCCCCCT',
            'AGCACAACACGG', 'AACACACACAGA', 'ACCACGATCCTC', 'ACACCACACCAC', 'GCAGCGTGACTT',
            'AACACGGCCCTC', 'TTTTAAAATTTT', 'CCAACCACAGTG', 'TCTCTTGGACCC'
            ]

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

nb_pct = np.zeros((len(seq_string_list), 4))
for (i,j) in zip(np.arange(len(seq_string_list)), seq_string_list):
    for k in j:
        if k == 'A':
            nb_pct[i][0] += 1/len(j)
        elif k == 'C':
            nb_pct[i][1] += 1/len(j)
        elif k == 'G':
            nb_pct[i][2] += 1/len(j)
        else: #k == 'T'
            nb_pct[i][3] += 1/len(j)

mw_list = []
for seq_string in seq_string_list:
    seq = Seq(seq_string)
    mw_list.append(molecular_weight(seq))

dg_list = []
for seq in seq_string_list:
    dg_list.append(dg(seq))

melting_temp = {'Sequence': seq_string_list,
                'Tm_Wallace': mt_wallace_list,
                'Tm_GC': mt_GC_list,
                'Tm_Breslauer': mt_breslauer_list,
                'Tm_Sugimoto': mt_sugimoto_list}

base_content = {'Sequence': seq_string_list,
                'A Percentage': nb_pct[:, 0],
                'C Percentage': nb_pct[:, 1],
                'G Percentage': nb_pct[:, 2],
                'T Percentage': nb_pct[:, 3]}

mw = {'Sequence': seq_string_list,
        'Molecular weight': mw_list}


#export DataFrame to text file
#with open('TmInvestigation.txt', 'a') as f:
#    df_string = df.to_string(header=True, index=False)
#    f.write(df_string)

dispersion = [
            1,1,1,1,1,
            0.5,1,1,1,1,
            0.5,1,1,0.5,0.5,
            0.5,1,1,1,1,
            1,1,1,0.5,
            0.5,0.5,
            0,0,0,0,0,
            0,0,0,0
            ]

df = pd.DataFrame({'Dispersion': dispersion,
                    'Tm_Wallace': mt_wallace_list,
                    'Tm_GC': mt_GC_list,
                    'Tm_Breslauer': mt_breslauer_list,
                    'Tm_Sugimoto': mt_sugimoto_list,
                    'Molecular weight': mw_list,
                    'A': nb_pct[:, 0],
                    'C': nb_pct[:, 1],
                    'G': nb_pct[:, 2],
                    'T': nb_pct[:, 3],
                    'Structure': dg_list})



df_corr = df.corr()
plt.figure(figsize = (10,7))
sns.set(font_scale=1)
ax = sns.heatmap(df_corr, annot=True)
ax.figure.tight_layout()
plt.show()