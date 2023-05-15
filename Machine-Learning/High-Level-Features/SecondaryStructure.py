from seqfold import fold, dg, dg_cache, dot_bracket
import pandas as pd
import numpy as np

seq_string_list = ['TATTATTATTAT', 'TTATTTTTTATT', 'ATTTTTTTTTTA', 'TTATATTATATT', 'CCACCACCACCA',
            'GCGCGCGCGCGC', 'TGTGTTTTGTGT', 'TTTTCCCCCCCC', 'TTTCCCCCCCCC', 'CCTCCCCCCCCT',
            'GTGTGTGTGTGT', 'ATTTATTTATTT', 'CCCCCTCCCCCT', 'TTTCCCCCCCCT', 'CCTTTCCCCCCT',
            'GGGGTTTTGGGG', 'AACACCACACAC', 'TTTTCCCCCCCT', 'ACACACACAACG', 'AACACACCACAG',
            'ACACACCATCAG', 'AGCAGCACACGA', 'ACGCCAACACAT', 'AACACACACAGC', 'AGCACCAGACAG',
            'ACGCACCGACAG']

dg_list = []
for seq in seq_string_list:
    print(seq)
    structs = fold(seq)
    for struct in structs:
        print(struct)

    print(dot_bracket(structs))
    print('')
    dg_list.append(dg(seq))

dg_list = np.array(dg_list)
#print(dg_list)
dg_list[np.isneginf(dg_list)] = -1e10
dg_list[np.isinf(dg_list)] = 1e10

secondary_structure = {'Sequence': seq_string_list,
                        'delta_G': dg_list}

df = pd.DataFrame(secondary_structure)
'''
#export DataFrame to text file
with open('SecondaryStructure.txt', 'a') as f:
    df_string = df.to_string(header=True, index=False)
    f.write(df_string)
'''
