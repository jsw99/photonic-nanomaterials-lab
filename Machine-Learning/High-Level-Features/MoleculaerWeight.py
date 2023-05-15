import numpy as np
import pandas as pd

seq_string_list = ['TATTATTATTAT', 'TTATTTTTTATT', 'ATTTTTTTTTTA', 'TTATATTATATT', 'CCACCACCACCA',
            'GCGCGCGCGCGC', 'TGTGTTTTGTGT', 'TTTTCCCCCCCC', 'TTTCCCCCCCCC', 'CCTCCCCCCCCT',
            'GTGTGTGTGTGT', 'ATTTATTTATTT', 'CCCCCTCCCCCT', 'TTTCCCCCCCCT', 'CCTTTCCCCCCT',
            'GGGGTTTTGGGG', 'AACACCACACAC', 'TTTTCCCCCCCT', 'ACACACACAACG', 'AACACACCACAG',
            'ACACACCATCAG', 'AGCAGCACACGA', 'ACGCCAACACAT', 'AACACACACAGC', 'AGCACCAGACAG',
            'ACGCACCGACAG']

mw_list = [3623.04, 3604.66, 3604.66, 3623.04, 3504.32,
        3648.38, 3687.04, 3467.56, 3452.72, 3437.88,
        3737.42, 3613.85, 3437.88, 3467.56, 3467.56,
        3787.8, 3552.38, 3482.4, 3592.41, 3592.41,
        3583.22, 3648.44, 3583.22, 3592.41, 3648.44,
        3624.41]

df = pd.DataFrame({'Sequence': seq_string_list, 
                    'Molecular Weight': mw_list})

#print(df)

#export DataFrame to text file
with open('MolecularWeight.txt', 'a') as f:
    df_string = df.to_string(header=True, index=False)
    f.write(df_string)