
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.io import fits
import scipy
from time import time

### Load the file with the dataset ###
hdulist = fits.open('/Users/coronado/Documents/Gaia_DR2/actions_angles_GaiaDR2_Schoenrich_3kpc_01.fits')
tbdata = hdulist[1]
tbdata.data.shape
tbdata.header
tbdata = hdulist[1].data # assume the first extension is a table


JR = tbdata.field("JR_kpckms")
Jz = tbdata.field("Jz_kpckms")
Lz = tbdata.field("Lz_kpckms")
TR = tbdata.field("TR_rad")
TP = tbdata.field("TP_rad")
TZ = tbdata.field("TZ_rad")

### Ignore stars with "crazy values for the actions (JR,Jz,Lz) and angles (TR,TZ,TP) ###
idx_in = np.where((JR<9000*8*220)&(Jz<9000*8*220)&(Lz<9000*8*220)&(np.isfinite(TP))&(np.isfinite(TZ))&(np.isfinite(TR))&(TP<9999.99) &(TR<9999.99) &(TZ<9999.99))

Jr = tbdata.field("JR_kpckms")[idx_in]
Jz = tbdata.field("Jz_kpckms")[idx_in]
Lz = tbdata.field("Lz_kpckms")[idx_in]
TR = tbdata.field("TR_rad")[idx_in]
TP = tbdata.field("TP_rad")[idx_in]
TZ = tbdata.field("TZ_rad")[idx_in]

### Load file that has the pairs information (file that comes from metric--pairs) ###
pairs = np.loadtxt('pairs_GDR2_RVS_3kpc_01_first_bin_RVS.txt')
primary  = pairs[:,0]
secondary = pairs[:,1]
primary = primary.astype(int)
secondary = secondary.astype(int)

dist_jR = (1/np.var(Jr))*((Jr[primary] - Jr[secondary])**2)
dist_jz = (1/np.var(Jz))*((Jz[primary]- Jz[secondary])**2)
dist_lz = (1/np.var(Lz))*((Lz[primary] - Lz[secondary])**2)
dist =    dist_jR + dist_jz + dist_lz

diff_tR = np.fabs(TR[primary] - TR[secondary])
diff_tz = np.fabs(TZ[primary]-TZ[secondary])
diff_tp = np.fabs(TP[primary]-TP[secondary])

diff_tR[diff_tR>np.pi] = 2*np.pi - diff_tR[diff_tR>np.pi]
diff_tz[diff_tz>np.pi] = 2*np.pi - diff_tz[diff_tz>np.pi]
diff_tp[diff_tp>np.pi] = 2*np.pi - diff_tp[diff_tp>np.pi]


dist_tR = (1/2.95)*diff_tR**2
dist_tz = (1/2.95)*diff_tz**2
dist_tp = (1/0.03)*diff_tp**2

metric_J_angles = dist + (dist_tR + dist_tz + dist_tp)
metric_J_angles = np.sqrt(metric_J_angles)/np.sqrt(6)

### Choose the linking length ###
L = 10**(-1.85) ## This gives me X groups ##
idx = np.where((metric_J_angles <=L))[0]


primary_new = primary[idx]
secondary_new = secondary[idx]
metric_J_angles_new = metric_J_angles[idx]

print(len(secondary_new))

values = np.unique(primary_new)

secondary_new_array = []
primary_new_array = []


cg_save = []
og_save = []
primary_save = []
primary_other_save = []
t_in = time()

for v in values:
    ig = np.where(primary_new == v)[0]
    secondary_new_array.append(secondary_new[ig])
    primary_new_array.append(v)
Np = len(primary_new_array)
id_primary = Np*[[],]

for i in range(len(secondary_new_array)-1):
    current_group = secondary_new_array[i]
    other_groups = secondary_new_array[i+1:]
    primary = primary_new_array[i]
    primary_other = primary_new_array[i+1:]
    aux = []
    for cg in current_group:
        for k in range(len(other_groups)):
            og = other_groups[k]
            #            #if not set(og).isdisjoint(cg):
            for s in og:
                if cg==s:
                    knew = k+1+i ## k is a local variable ##
                    if knew in aux: # avoids repetition for the primary star #
                        continue
                    aux.append(knew)
    id_primary[i] = aux # list of primaries connected to this star ##


## Recorre los indices id_primary, busca hacia donde apunta y los junta en una lista
## This goes through the indexes id_primary, searches them and puts them together in a list ###
def find(p, lp, idp, skip): ## it takes two parameters: the primary and a list where it keeps the group's info ##
    for i in idp[p]:
        if not skip[i]:
            lp+= find(i,[i], idp, skip)
    return(lp)

id_primary2 = Np*[[],] ## the other "primaries" pointing from the "other direction" ##
for p in range(Np):
    for ii in id_primary[p]:
        if len(id_primary2[ii])>0:
            id_primary2[ii]+= [p]
        else:
            id_primary2[ii] = [p]


t0 = time()
skip = np.zeros(Np, dtype=bool)
group_fin = []
for p in range(Np):
    if skip[p]:
        continue
    Ngroup,Naux = 1,1
    group = [p]
    stars = id_primary[p]
    while True:
        group += stars
        group = list(np.unique(group))
        Ngroup = len(group)
        if Ngroup == Naux:
            break
        Naux = Ngroup
        aux = []
        for s in stars:
            if not skip[s]:
                skip[s] = True
                aux += id_primary[s]
                aux += id_primary2[s]
        stars = list(np.unique(aux))
    skip[group] = True
    if len(group)>1:
        group_fin.append(group)


group_f = []
for i in range(len(group_fin)):
    aux = []
    for j in group_fin[i]:
        aux.append(primary_new_array[j])
        aux+=(list(secondary_new_array[j]))
    group_f.append(aux)


## this eliminates repeated elements within a same group ##
Group_final = []
for i in range(len(group_f)):
    u, c = np.unique(group_f[i], return_counts=True)
    dup = u[c > 1]
    Group_final.append(u)


ngr =0
nmin = 20
#nmin = 50
for i in range(len(Group_final)):
    if len(Group_final[i])>=nmin:
        #if len(Group_final[i])>=nmin and (len(Group_final[i])<15):
        print(Group_final[i],i, len(Group_final[i]))
        ngr+= 1

print(ngr)


id_split = Group_final
N_group_split = []
N_id_split = []
for i in range(len(id_split)):
    if (len(Group_final[i])>=nmin): #and (len(Group_final[i])<15) :
        N_group_split.append(len(id_split[i]))
        N_id_split.append(id_split[i])
N_group_split = np.asarray(N_group_split)


id_G1 = Group_final
N_group_G1 = []
for i in range(len(id_G1)):
    N_group_G1.append(len(id_G1[i]))
N_group_G1 = np.asarray(N_group_G1)
#G1 = np.save('G1_Lz_065.npy', N_group_G1)
id_G1 = np.save('id_Link_1_85', id_G1)
