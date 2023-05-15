import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sam_scores = [3.93196383, 21.70206373, 8.36853388, 0.95796721, 14.29305771, 4.41929825,
                18.62884611, 13.55995214, 19.09643357, 15.05841842, 6.4185908, 2.53986694,
                4.66475473, 7.85644898, 9.84625476, 16.33406361, 6.93474251, 12.18638009,
                53.73858215, 6.58886888, 14.94425551, 15.09345632, 7.38712673, 18.23156387,
                7.52867457, 5.95982536]

#df = pd.read_csv('SecondaryStructure.txt', header=None, delimiter='\s+', skiprows=1, names=['Sequence', 'delta_G'])
#print(df)
#dg = df['delta_G']
tm_wallace = np.loadtxt('MeltingTemperature.txt', usecols=1, skiprows=1)
tm_gc = np.loadtxt('MeltingTemperature.txt', usecols=2, skiprows=1)
tm_breslauer = np.loadtxt('MeltingTemperature.txt', usecols=3, skiprows=1)
tm_sugimoto = np.loadtxt('MeltingTemperature.txt', usecols=4, skiprows=1)
mw = np.loadtxt('MolecularWeight.txt', usecols=1, skiprows=1)
a_content = np.loadtxt('BaseContent.txt', usecols=1, skiprows=1)
c_content = np.loadtxt('BaseContent.txt', usecols=2, skiprows=1)
g_content = np.loadtxt('BaseContent.txt', usecols=3, skiprows=1)
t_content = np.loadtxt('BaseContent.txt', usecols=4, skiprows=1)
sec_struct = np.loadtxt('SecondaryStructure.txt', usecols=1, skiprows=1)

df = pd.DataFrame({'SAM': sam_scores,
                    'Tm Wallace': tm_wallace,
                    'Tm GC': tm_gc,
                    'Tm Breslauer': tm_breslauer,
                    'Tm Sugimoto': tm_sugimoto,
                    'Molecular Weight': mw,
                    'A': a_content,
                    'C': c_content,
                    'G': g_content,
                    'T': t_content,
                    'Structure': sec_struct})

df_corr = df.corr()
plt.figure(figsize = (10,7))
sns.set(font_scale=1)
ax = sns.heatmap(df_corr, annot=True)
ax.figure.tight_layout()
plt.show()

r_wallace = np.corrcoef(tm_wallace, sam_scores)
print(r_wallace)
r_gc = np.corrcoef(tm_gc, sam_scores)
print(r_gc)
r_breslauer = np.corrcoef(tm_breslauer, sam_scores)
print(r_breslauer)
r_sugimoto = np.corrcoef(tm_sugimoto, sam_scores)
print(r_sugimoto)
r_mw = np.corrcoef(mw, sam_scores)
print(r_mw)
r_acont = np.corrcoef(a_content, sam_scores)
print(r_acont)
r_ccont = np.corrcoef(c_content, sam_scores)
print(r_ccont)
r_gcont = np.corrcoef(g_content, sam_scores)
print(r_gcont)
r_tcont = np.corrcoef(t_content, sam_scores)
print(r_tcont)
r_struct = np.corrcoef(sec_struct, sam_scores)
print(r_struct)


'''
fig, ax = plt.subplots(nrows=2, ncols=5, sharey=True, figsize=(11,5))

ax1 = ax[0][0]
ax1.plot(tm_wallace, sam_scores, 'ro')
ax1.set_xlabel('T$_{m}$ Wallace')
ax1.xaxis.set_tick_params(which='major', top='on', direction='in')
ax1.yaxis.set_tick_params(which='major', top='on', direction='in')

ax2 = ax[0][1]
ax2.plot(tm_gc, sam_scores, 'ro')
ax2.set_xlabel('T$_{m}$ GC')
ax2.xaxis.set_tick_params(which='major', top='on', direction='in')
ax2.yaxis.set_tick_params(which='major', top='on', direction='in')

ax3 = ax[0][2]
ax3.plot(tm_breslauer, sam_scores, 'ro')
ax3.set_xlabel('T$_{m}$ Breslauer')
ax3.xaxis.set_tick_params(which='major', top='on', direction='in')
ax3.yaxis.set_tick_params(which='major', top='on', direction='in')

ax4 = ax[0][3]
ax4.plot(tm_sugimoto, sam_scores, 'ro')
ax4.set_xlabel('T$_{m}$ Sugimoto')
ax4.xaxis.set_tick_params(which='major', top='on', direction='in')
ax4.yaxis.set_tick_params(which='major', top='on', direction='in')

ax5= ax[0][4]
ax5.plot(mw, sam_scores, 'bo')
ax5.set_xlabel('Molecular Weight')
ax5.xaxis.set_tick_params(which='major', top='on', direction='in')
ax5.yaxis.set_tick_params(which='major', top='on', direction='in')

ax6 = ax[1][0]
ax6.plot(a_content, sam_scores, 'mo')
ax6.set_xlabel('A Percentage')
ax6.xaxis.set_tick_params(which='major', top='on', direction='in')
ax6.yaxis.set_tick_params(which='major', top='on', direction='in')

ax7 = ax[1][1]
ax7.plot(c_content, sam_scores, 'mo')
ax7.set_xlabel('C Percentage')
ax7.xaxis.set_tick_params(which='major', top='on', direction='in')
ax7.yaxis.set_tick_params(which='major', top='on', direction='in')

ax8 = ax[1][2]
ax8.plot(g_content, sam_scores, 'mo')
ax8.set_xlabel('G Percentage')
ax8.xaxis.set_tick_params(which='major', top='on', direction='in')
ax8.yaxis.set_tick_params(which='major', top='on', direction='in')

ax9 = ax[1][3]
ax9.plot(t_content, sam_scores, 'mo')
ax9.set_xlabel('T Percentage')
ax9.xaxis.set_tick_params(which='major', top='on', direction='in')
ax9.yaxis.set_tick_params(which='major', top='on', direction='in')

ax10 = ax[1][4]
ax10.plot(sec_struct, sam_scores, 'co')
ax10.set_xlabel('∆G')
ax10.xaxis.set_tick_params(which='major', top='on', direction='in')
ax10.yaxis.set_tick_params(which='major', top='on', direction='in')

fig.suptitle('Correlation between spectral angle and different features')
fig.text(0.04, 0.5, 'Spectral Angle (degree)', va='center', rotation='vertical')
fig.tight_layout(rect=[0.07, 0.07, 1, 1])
plt.show()
'''

