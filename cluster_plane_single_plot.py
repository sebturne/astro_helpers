import astro_helpers as H
import numpy as N
import numpy.random as NR
import matplotlib as M
M.use('Agg')
import matplotlib.pyplot as MP
from sklearn.neighbors import KernelDensity as SNK
from sklearn.model_selection import GridSearchCV as SG

MP.rcParams.update({'figure.figsize':[4,4], 'figure.dpi': 300, 'font.family': 'Times New Roman', 'font.size': 10, 'text.usetex': True, 'figure.autolayout': True})

feat_set = 'mstar_smdust_ssfr_hlr_bt' # features
k = 7 # number of clusters
p = [0,4] # features to plot

data = N.genfromtxt('Pre/' + feat_set + '_tl.txt', delimiter = ',') # feature data
lbls = N.genfromtxt('../../PhD/Data/Clustering/EAGLE/RefL0100N1504_z_0/Post/Labels/' + feat_set + '_tln_lbls_k%i.txt' % k, delimiter = ',') # all labels at k
phi  = N.genfromtxt('Post/Clustering/' + feat_set + '_tln_staco.txt', delimiter = ',')[:,1:] # clustering quality-of-fit data
phi  = phi[phi[:,1] == k,:] # clustering quality-of-fit data at k

best = N.where(phi[:,0] == min(phi[:,0]))[0][0] # best-fitting clustering outcome at k
clst = lbls[:,best].reshape(-1,1) # getting labels

# plot metadata (dependent on features plotted)
feat_lbl = ['$\log_{10}(M_{*}/\mathrm{M_{\odot}})$', '$\log_{10}(M_{d}/M_{*})$', '$log_{10}(SSFR/\mathrm{yr^{-1}})$', '$log_{10}(R_{1/2,l}/\mathrm{kpc})$', '$B/T_{l}$']
feat_lim = [[9.5, 11.5], [-5.5, -1.0], [-13.50, -8.75], [0.0, 1.5], [0, 1]]
feat_tix = [[10, 11], [-5, -4, -3, -2], [-13, -12, -11, -10, -9], [0.25, 0.75, 1.25], [0.0, 0.5, 1.0]]
col = ['#ee7722','#975a09','#ee3333','#66aa55','#992288','#9370db','#3366aa'][::-1]
leg = [M.patches.Patch(color = col[i], label = cnm[i]) for i in range(k)]

cnm = ['E' + str(i) for i in range (7,0,-1)][::-1] # cluster names
cod = H.FS(data[:,2], clst[:,0])[::-1] # sorting clusters by ssfr

kde = N.zeros((25,25,k))

fig, ax = MP.subplots()

ax.scatter(data[:,p[0]], data[:,p[1]], s = 1.2, c = 'k', linewidths = 0) # plotting base feature data

for i in range(k):
    x,y = N.mgrid[N.min(data[:,p[0]]):N.max(data[:,p[0]]):25j, N.min(data[:,p[1]]):N.max(data[:,p[1]]):25j]
    grd = N.vstack([x.ravel(), y.ravel()]).T # setting up for grid search
    
    src = SG(SNK(kernel = 'gaussian'), {'bandwidth': N.logspace(-2, 0, 6)}, cv = 5) # grid search for gaussian kde
    src.fit(data[clst[:,0] == cod[i]][:,p])

    knl = src.best_estimator_ # best-fitting gaussian kde
    knl = N.exp(knl.score_samples(grd)) # predicting
    knl = knl / N.amax(knl, axis = 0) # normalising
    kde[:,:,i] = knl.reshape(25,25)
    
for i in range(k):
    cx = N.mean(data[clst[:,0] == cod[i]][:,p[0]]) # cluster means on x-axis
    cy = N.mean(data[clst[:,0] == cod[i]][:,p[1]]) # cluster means on y-axis
    ax.contour(x, y, kde[:,:,cod[i]], [0.33], colors = 'w', linewidths = 5) # cluster contours at 33%
    ax.contour(x, y, kde[:,:,cod[i]], [0.33], colors = col[cod[i]], linewidths = 3) # cluster contours at 33%
    ax.scatter(cx, cy, s = 100, c = col[i], edgecolor = 'w', lw = 1, zorder = 3)

ax.set(xticks = feat_tix[p[0]], yticks = feat_tix[p[1]], xlim = feat_lim[p[0]], ylim = feat_lim[p[1]], xlabel = feat_lbl[p[0]], ylabel = feat_lbl[p[1]])

fig.savefig('Post/Plane/' + feat_set + '_tln_k%i_%i%i.png' % (k,p[0],p[1]))
