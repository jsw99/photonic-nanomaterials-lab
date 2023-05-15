import numpy as np
import pandas as pd

seq_string_list = ['TATTATTATTAT', 'TTATTTTTTATT', 'ATTTTTTTTTTA', 'TTATATTATATT', 'CCACCACCACCA',
            'GCGCGCGCGCGC', 'TGTGTTTTGTGT', 'TTTTCCCCCCCC', 'TTTCCCCCCCCC', 'CCTCCCCCCCCT',
            'GTGTGTGTGTGT', 'ATTTATTTATTT', 'CCCCCTCCCCCT', 'TTTCCCCCCCCT', 'CCTTTCCCCCCT',
            'GGGGTTTTGGGG', 'AACACCACACAC', 'TTTTCCCCCCCT', 'ACACACACAACG', 'AACACACCACAG',
            'ACACACCATCAG', 'AGCAGCACACGA', 'ACGCCAACACAT', 'AACACACACAGC', 'AGCACCAGACAG',
            'ACGCACCGACAG']

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


base_content = {'Sequence': seq_string_list,
                'A Percentage': nb_pct[:, 0],
                'C Percentage': nb_pct[:, 1],
                'G Percentage': nb_pct[:, 2],
                'T Percentage': nb_pct[:, 3]}

df = pd.DataFrame(base_content)
print(df)

#export DataFrame to text file
with open('BaseContent.txt', 'a') as f:
    df_string = df.to_string(header=True, index=False)
    f.write(df_string)