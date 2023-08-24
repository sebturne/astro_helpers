import astro_helpers as H
import numpy as N
import matplotlib as M
M.use('Agg')
import matplotlib.pyplot as MP
from sklearn.neighbors import KernelDensity as SNK
from sklearn.model_selection import GridSearchCV as SG

MP.rcParams.update({'figure.figsize':[8,8], 'figure.dpi': 300, 'font.family': 'Times New Roman', 'font.size': 10, 'text.usetex': True})

feat_set = 'mstar_smdust_ssfr_hlr_bt' # features
k = 7 # number of clusters

data = N.genfromtxt('Pre/' + feat_set + '_tl.txt', delimiter = ',') # feature data
lbls = N.genfromtxt('../../PhD/Data/Clustering/EAGLE/RefL0100N1504_z_0/Post/Labels/' + feat_set + '_tln_lbls_k%i.txt' % k, delimiter = ',') # all labels at k
phi  = N.genfromtxt('Post/Clustering/' + feat_set + '_tln_staco.txt', delimiter = ',')[:,1:] # clustering quality-of-fit data
phi  = phi[phi[:,1] == k,:] # clustering quality-of-fit data at k

best = N.where(phi[:,0] == min(phi[:,0]))[0][0] # best-fitting clustering outcome at k
clst = lbls[:,best].reshape(-1,1) # getting labels

# plot metadata (for all five features)
feat_lbl = ['$\log_{10}(M_{*}/\mathrm{M_{\odot}})$', '$\log_{10}(sM_{d})$', '$log_{10}(sSFR/\mathrm{yr^{-1}})$', '$log_{10}(R_{1/2,m}/\mathrm{kpc})$', '$B/T_{m}$']
feat_lim = [[9.5, 11.5], [-5.25, -1.5], [-13.25, -8.75], [0, 1.25], [0, 1]]
feat_tix = [[10, 11], [-5, -4, -3, -2], [-13, -12, -11, -10, -9], [0.2, 0.6, 1.0], [0.0, 0.5, 1.0]]
col = ['#ee7722','#975a09','#ee3333','#66aa55','#992288','#9370db','#3366aa'][::-1]
leg = [M.patches.Patch(color = col[i], label = cnm[i]) for i in range(k)]

cnm = ['E' + str(i) for i in range (7,0,-1)][::-1] # cluster names
cod = H.FS(data[:,2], clst[:,0])[::-1] # sorting clusters by ssfr

d = data.shape[1]

fig = MP.figure()

for c in range(d): # iterates over all features
    for r in range(d): # also iterates over all features
        if c <  r: # only plotting if below diagonal
            ax = MP.subplot2grid((d,d), (r,c))
            ax.scatter(data[:,c], data[:,r], s = 0.3, c = '#000000', linewidths = 0) # plotting base feature data in current panel

            kde = N.zeros((25,25,k))

            for i in range(k): # grid search for best-fitting gaussian kde in current panel
                x,y = N.mgrid[N.min(data[:,c]):N.max(data[:,c]):25j, N.min(data[:,r]):N.max(data[:,r]):25j]
                grd = N.vstack([x.ravel(), y.ravel()]).T
                
                src = SG(SNK(kernel = 'gaussian'), {'bandwidth': N.logspace(-2, 0, 6)}, cv = 5)
                src.fit(data[clst[:,0] == cod[i]][:,[c,r]])

                knl = src.best_estimator_
                knl = N.exp(knl.score_samples(grd))
                knl = knl / N.amax(knl, axis = 0)
                kde[:,:,i] = knl.reshape(25,25)

            for i in range(k): # plotting clusters based on best-fitting gaussian kde
                cx = N.mean(data[clst[:,0] == cod[i]][:,c])
                cy = N.mean(data[clst[:,0] == cod[i]][:,r])
                ax.contour(x, y, kde[:,:,i], [0.33], colors = '#ffffff', linewidths = 2.5, zorder = 3)
                ax.contour(x, y, kde[:,:,i], [0.33], colors = col[i], linewidths = 1.75,zorder = 3)
                ax.scatter(cx, cy, s = 25, c = col[i], edgecolor = '#ffffff', lw = 0.375, zorder = 4)

            ax.set(xlabel = feat_lbl[c], ylabel = feat_lbl[r], xlim = (feat_lim[c][0], feat_lim[c][1]), ylim = (feat_lim[r][0], feat_lim[r][1]), xticks = feat_tix[c], yticks = feat_tix[r])
            if c != 0:
                ax.axes.get_yaxis().set_visible(False)
            if r != d-1:
                ax.axes.get_xaxis().set_visible(False)
        if c == r: # plotting histograms along diagonal
            ax = MP.subplot2grid((d,d), (r,c))
            for i in range(k):
                ax.hist(data[clst[:,0] == cod[i]][:,c], bins = 'fd', histtype = 'step', color = '#ffffff', lw = 2.0)
                ax.hist(data[clst[:,0] == cod[i]][:,c], bins = 'fd', histtype = 'step', color = col[i], lw = 1.25)
            ax.axes.get_yaxis().set_visible(False)
            ax.set(xlabel = feat_lbl[c], xlim = (feat_lim[c][0], feat_lim[c][1]), xticks = feat_tix[c])
            if r <  d-1:
                ax.axes.get_xaxis().set_visible(False)
        if c == d-1 and r == 0: # plotting legend
            ax = MP.subplot2grid((d,d), (r,c))
            ax.legend(handles = leg, loc = 'upper center', fontsize = 10, frameon = True, facecolor = '#ffffff', edgecolor = '#000000', framealpha = 1.0, fancybox = True, borderpad = 0.5)
            ax.axes.get_yaxis().set_visible(False)
            ax.axes.get_xaxis().set_visible(False)
            ax.axis('off')

fig.subplots_adjust(wspace = 0.05, hspace = 0.05)
fig.savefig('Post/Corner/' + feat_set + '_tln_corner_k%i.png' % k, bbox_inches = 'tight')
