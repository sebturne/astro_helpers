import astro_helpers
import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot
import sklearn.metrics

matplotlib.pyplot.rcParams.update({'figure.figsize':[4,4], 'figure.dpi': 300, 'font.family': 'Times New Roman', 'font.size': 10, 'text.usetex': True})

feat_set = 'mstar_n_ssfr_sig5'
k = [2,3,6,9]

zscr = numpy.genfromtxt('Pre/' + feat_set + '_tlz.txt', delimiter = ',')
lbls = numpy.genfromtxt('../../Data/Clustering2/' + feat_set + '_tlz_lbls_k%i.txt' % k[0], delimiter = ',')
phi  = numpy.genfromtxt('../../Data/Clustering2/' + feat_set + '_tlz_staco.txt', delimiter = ',')[:,1:]
phi  = phi[phi[:,1] == k[0],:]

best = numpy.where(phi[:,0] == min(phi[:,0]))[0][0]
clst = lbls[:,best].reshape(-1,1)

n = [0] * (k[0])

for j in range(k[0]):
    n[j] = len(clst[clst[:,0] == j,0])

ma = astro_helpers.FS(zscr[:,1], clst[:,0])

ya = N.arange(0.5-(float(k[0])/2),(float(k[0])/2),1)

fig, ax = matplotlib.pyplot.subplots()

for j in range(k[0]):
    ax.text(0, ya[j], '$%i$' % (n[ma[j]]), ha = 'center', va = 'center', bbox = dict(facecolor = 'w', edgecolor = 'k', boxstyle = 'round,pad=0.5'))

ax.text(0, ya[j] + 0.75, '$k = %i$' % k[0], ha = 'center', va = 'center')

for i in range(1,len(k)):

    lbls = numpy.genfromtxt('../../Data/Clustering2/' + feat_set + '_tlz_lbls_k%i.txt' % k[i], delimiter = ',')
    phi = numpy.genfromtxt('../../Data/Clustering2/' + feat_set + '_tlz_staco.txt', delimiter = ',')[:,1:]
    phi = phi[phi[:,1] == k[i],:]
    
    best = numpy.where(phi[:,0] == min(phi[:,0]))[0][0]
    clst = numpy.hstack((clst,lbls[:,best].reshape(-1,1)))

    yb = numpy.arange(0.5-(float(k[i])/2),(float(k[i])/2),1)
    mb = astro_helpers.FS(zscr[:,2], clst[:,i])

    coma = sklearn.metrics.confusion_matrix(clst[:,i-1],clst[:,i])
    coma = coma[:(k[i-1]), :(k[i])]
    coma = coma[[x for x in ma],:]
    coma = coma[:,[x for x in mb]]
    
    n = [0] * k[i]

    for j in range(k[i]):
        n[j] = len(clst[clst[:,i] == j,i])

    for l in range(k[i-1]):
        for j in range(k[i]):
            if k[i] < k[i-1]:
                o = ((float(coma[l,j]) / float(sum(coma[l,:])))**(1))
            else:
                o = ((float(coma[l,j]) / float(sum(coma[:,j])))**(1))
            ax.arrow(i-1, ya[l], 1, yb[j] - ya[l], fc = 'r', ec = 'r', alpha = o)

    for j in range(k[i]):
        ax.text(i, yb[j], '$%i$' % (n[mb[j]]), ha = 'center', va = 'center', bbox = dict(facecolor = 'w', edgecolor = 'k', boxstyle ='round,pad=0.5'))

    ax.text(i, yb[j] + 0.75, '$k = %i$' % k[i], ha = 'center', va = 'center')
    
    ya = yb
    ma = mb

ax.plot([0,1,2,3],[0,0.5,0.5,1],'k--')

ax.set(xticks = [], yticks = [], xlim = (-0.5,len(k)-0.5), ylim = (-4,4))
ax.axis('off')
fig.savefig('Post/Clustering/' + feat_set + '_tlz_hierarchy.png', bbox_inches = "tight", pad_inches=0)
