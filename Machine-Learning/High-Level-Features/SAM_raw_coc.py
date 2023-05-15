import sys
sys.path.append(r'/Users/jasonwang/L327/ML')
import spectral_angle_mapper
import matplotlib.pyplot as pyplot
import numpy as np

TAT4_raw = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/(TAT)4-HiPco_new.txt', usecols=2)
TAT4_coc = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/(TAT)4-HiPco+coc_new.txt', usecols=2)

TTAT6ATT_raw = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/TTAT6ATT-HiPco_new.txt', usecols=2)
TTAT6ATT_coc = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/TTAT6ATT-HiPco+coc_new.txt', usecols=2)

AT10A_raw = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/AT10A-HiPco_new.txt', usecols=2)
AT10A_coc = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/AT10A-HiPco+coc_new.txt', usecols=2)

TTATAT2ATT_raw = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/TTA(TAT)2ATT-HiPco_new.txt', usecols=2)
TTATAT2ATT_coc = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/TTA(TAT)2ATT-HiPco+coc_new.txt', usecols=2)

CCA4_raw = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/(CCA)4-HiPco_new.txt', usecols=2)
CCA4_coc = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/(CCA)4-HiPco+coc_new.txt', usecols=2)

GC6_raw = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/(GC)6-HiPco_new.txt', usecols=2)
GC6_coc = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/(GC)6-HiPco+coc_new.txt', usecols=2)

TG2T4GT2_raw = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/(TG)2T4(GT)2-HiPco_new.txt', usecols=2)
TG2T4GT2_coc = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/(TG)2T4(GT)2-HiPco+coc_new.txt', usecols=2)

T4C8_raw = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/T4C8-HiPco_new.txt', usecols=2)
T4C8_coc = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/T4C8-HiPco+coc_new.txt', usecols=2)

T3C9_raw = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/T3C9-HiPco_new.txt', usecols=2)
T3C9_coc = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/T3C9-HiPco+coc_new.txt', usecols=2)

CCTC8T_raw = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/CCTC8T-HiPco_new.txt', usecols=2)
CCTC8T_coc = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/CCTC8T-HiPco+coc_new.txt', usecols=2)

GT6_raw = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/(GT)6-HiPco_new.txt', usecols=2)
GT6_coc = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/(GT)6-HiPco+coc_new.txt', usecols=2)

ATTT3_raw = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/(ATTT)3-HiPco_new.txt', usecols=2)
ATTT3_coc = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/(ATTT)3-HiPco+coc_new.txt', usecols=2)

C5TC5T_raw = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/C5TC5T-HiPco_new.txt', usecols=2)
C5TC5T_coc = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/C5TC5T-HiPco+coc_new.txt', usecols=2)

T3C8T_raw = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/T3C8T-HiPco_new.txt', usecols=2)
T3C8T_coc = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/T3C8T-HiPco+coc_new.txt', usecols=2)

C2T3C6T_raw = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/C2T3C6T-HiPco_new.txt', usecols=2)
C2T3C6T_coc = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/C2T3C6T-HiPco+coc_new.txt', usecols=2)

G4T4G4_raw = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/G4T4G4-HiPco_new.txt', usecols=2)
G4T4G4_coc = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/G4T4G4-HiPco+coc_new.txt', usecols=2)

A2CAC2AC3_raw = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/A2CAC2(AC)3-HiPco_new.txt', usecols=2)
A2CAC2AC3_coc = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/A2CAC2(AC)3-HiPco+coc_new.txt', usecols=2)

T4C7T_raw = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/T4C7T-HiPco_new.txt', usecols=2)
T4C7T_coc = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/T4C7T-HiPco+coc_new.txt', usecols=2)

ACA4ACG_raw = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/A(CA)4ACG-HiPco_new.txt', usecols=2)
ACA4ACG_coc = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/A(CA)4ACG-HiPco+coc_new.txt', usecols=2)

AAC3CA2G_raw = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/A(AC)3(CA)2G-HiPco_new.txt', usecols=2)
AAC3CA2G_coc = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/A(AC)3(CA)2G-HiPco+coc_new.txt', usecols=2)

ACA2C2ATCAG_raw = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/A(CA)2C2ATCAG-HiPco_new.txt', usecols=2)
ACA2C2ATCAG_coc = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/A(CA)2C2ATCAG-HiPco+coc_new.txt', usecols=2)

AGC2ACACGA_raw = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/(AGC)2ACACGA-HiPco_new.txt', usecols=2)
AGC2ACACGA_coc = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/(AGC)2ACACGA-HiPco+coc_new.txt', usecols=2)

ACGC2A2CA2T_raw = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/ACGC2A2(CA)2T-HiPco_new.txt', usecols=2)
ACGC2A2CA2T_coc = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/ACGC2A2(CA)2T-HiPco+coc_new.txt', usecols=2)

AAC4AGC_raw = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/A(AC)4AGC-HiPco_new.txt', usecols=2)
AAC4AGC_coc = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/A(AC)4AGC-HiPco+coc_new.txt', usecols=2)

AGCAC2AGACAG_raw = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/AGCAC2AGACAG-HiPco_new.txt', usecols=2)
AGCAC2AGACAG_coc = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/AGCAC2AGACAG-HiPco+coc_new.txt', usecols=2)

ACGCAC2GACAG_raw = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/ACGCAC2GACAG-HiPco_new.txt', usecols=2)
ACGCAC2GACAG_coc = np.loadtxt('/Users/jasonwang/L327/ML/training_data/y_train_new/ACGCAC2GACAG-HiPco+coc_new.txt', usecols=2)

sam_scores = []

sam_scores.append(spectral_angle_mapper.spectral_angle_mapper(TAT4_coc, TAT4_raw))
sam_scores.append(spectral_angle_mapper.spectral_angle_mapper(TTAT6ATT_coc, TTAT6ATT_raw))
sam_scores.append(spectral_angle_mapper.spectral_angle_mapper(AT10A_coc, AT10A_raw))
sam_scores.append(spectral_angle_mapper.spectral_angle_mapper(TTATAT2ATT_coc, TTATAT2ATT_raw))
sam_scores.append(spectral_angle_mapper.spectral_angle_mapper(CCA4_coc, CCA4_raw))
sam_scores.append(spectral_angle_mapper.spectral_angle_mapper(GC6_coc, GC6_raw))
sam_scores.append(spectral_angle_mapper.spectral_angle_mapper(TG2T4GT2_coc, TG2T4GT2_raw))
sam_scores.append(spectral_angle_mapper.spectral_angle_mapper(T4C8_coc, T4C8_raw))
sam_scores.append(spectral_angle_mapper.spectral_angle_mapper(T3C9_coc, T3C9_raw))
sam_scores.append(spectral_angle_mapper.spectral_angle_mapper(CCTC8T_coc, CCTC8T_raw))
sam_scores.append(spectral_angle_mapper.spectral_angle_mapper(GT6_coc, GT6_raw))
sam_scores.append(spectral_angle_mapper.spectral_angle_mapper(ATTT3_coc, ATTT3_raw))
sam_scores.append(spectral_angle_mapper.spectral_angle_mapper(C5TC5T_coc, C5TC5T_raw))
sam_scores.append(spectral_angle_mapper.spectral_angle_mapper(T3C8T_coc, T3C8T_raw))
sam_scores.append(spectral_angle_mapper.spectral_angle_mapper(C2T3C6T_coc, C2T3C6T_raw))
sam_scores.append(spectral_angle_mapper.spectral_angle_mapper(G4T4G4_coc, G4T4G4_raw))
sam_scores.append(spectral_angle_mapper.spectral_angle_mapper(A2CAC2AC3_coc, A2CAC2AC3_raw))
sam_scores.append(spectral_angle_mapper.spectral_angle_mapper(T4C7T_coc, T4C7T_raw))
sam_scores.append(spectral_angle_mapper.spectral_angle_mapper(ACA4ACG_coc, ACA4ACG_raw))
sam_scores.append(spectral_angle_mapper.spectral_angle_mapper(AAC3CA2G_coc, AAC3CA2G_raw))
sam_scores.append(spectral_angle_mapper.spectral_angle_mapper(ACA2C2ATCAG_coc, ACA2C2ATCAG_raw))
sam_scores.append(spectral_angle_mapper.spectral_angle_mapper(AGC2ACACGA_coc, AGC2ACACGA_raw))
sam_scores.append(spectral_angle_mapper.spectral_angle_mapper(ACGC2A2CA2T_coc, ACGC2A2CA2T_raw))
sam_scores.append(spectral_angle_mapper.spectral_angle_mapper(AAC4AGC_coc, AAC4AGC_raw))
sam_scores.append(spectral_angle_mapper.spectral_angle_mapper(AGCAC2AGACAG_coc, AGCAC2AGACAG_raw))
sam_scores.append(spectral_angle_mapper.spectral_angle_mapper(ACGCAC2GACAG_coc, ACGCAC2GACAG_raw))

print(np.rad2deg(sam_scores))
