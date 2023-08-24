import gama_functions as G
import numpy as N
import numpy.random as NR
import matplotlib as M
M.use('Agg')
import matplotlib.pyplot as MP
import sklearn.neighbors.kde as SNK
import sklearn.grid_search as SG

MP.rcParams.update({'figure.figsize':[8,8], 'figure.dpi': 300, 'font.family': 'Times New Roman', 'font.size': 10, 'text.usetex': True})

#########
# setup #
#########

data = N.genfromtxt('data.txt', delimiter = ',') # replace 'data.txt' with your data
clst = N.genfromtxt('labels.txt', delimiter = ',') # replace 'labels.txt' with your labels

# clst MUST have shape (xxx, 1), where xxx is the sample size, for this code to work
# use print(clst.shape) to see what shape it has
# if clst just reads in with shape (xxx,), then use clst = clst.reshape(-1,1) to fix it

d = N.shape(data)[1] # number of features (measured from number of columns in data)
k = N.unique(clst).shape[0] # number of clusters (measured from clst), haven't tested this fully so if it doesn't work just set it manually

lbls = ['$\log_{10}(M_{*}/$M$_{\odot}$', '', '', '', ''] # axis labels, fill these out with your own labels, add more if needed, one example included here
lims = [[9.5, 12.0], [,], [,], [,], [,]] # axis limits, as above
tiks = [[9.5,10.0,10.5,11.0,11.5], [,], [,], [,], [,]] # axis ticks, as above

# if you're not bothered about axis limits/ticks, i.e. if you just want them automatically set,
# then delete 'lims'/'tiks' above, and delete references to them in ax.set() further below
# ax.set() shows up in two places below, so make sure you catch them both if you do this

# setting up colours and cluster names here, replace with whatever you need
col = ['#3366aa', '#9370db', '#66aa55', '#ee7722', '#ee3333'] # cluster colours, hex codes
cnm = ['G' + str(i) for i in range(1,k+1)] # cluster names, i.e. ['G1', G2', ...]

leg = [M.patches.Patch(color = col[i], label = cnm[i]) for i in range(k)] # legend, based on colours and cluster names

############
# plotting #
############

fig = MP.figure() # setting up figure

for c in range(d): # columns
    for r in range(d): # rows
        if c <  r: # setting up tiles for scatter/contour plots
            ax = MP.subplot2grid((d,d), (r,c)) # setting up axes
            ax.set(xlabel = lbls[c], ylabel = lbls[r], xlim = (lims[c][0], lims[c][1]), ylim = (lims[r][0], lims[r][1]), xticks = tiks[c], yticks = tiks[r])
            
            ax.scatter(data[:,c], data[:,r], s = 0.3, c = '#000000', linewidths = 0) # plotting data as scatter points in background

            kde = N.zeros((25,25,k)) # setting up stack of kernel density estimates for k clusters

            x,y = N.mgrid[N.min(data[:,c]):N.max(data[:,c]):25j, N.min(data[:,r]):N.max(data[:,r]):25j]
            grd = N.vstack([x.ravel(), y.ravel()]).T
            src = SG.GridSearchCV(SNK.KernelDensity(kernel = 'gaussian'), {'bandwidth': N.logspace(-2, 0, 6)}, cv = 5) # setting up grid search for best fitting kde
            
            for i in range(k): # calculating kde for each cluster
                src.fit(data[clst[:,0] == i][:,[c,r]]) # searching

                knl = src.best_estimator_ # selecting best fitting kde
                knl = N.exp(knl.score_samples(grd)) # best fitting kde in linear units
                knl = knl / N.amax(knl, axis = 0) # normalising best fitting kde to max of 1
                kde[:,:,i] = knl.reshape(25,25) # adding to stack

            for i in range(k): # plotting clusters
                cx, cy = N.mean(data[clst[:,0] == i][:,c], data[clst[:,0] == i][:,r]) # cluster centroid
                ax.contour(x, y, kde[:,:,i], [0.25], colors = '#ffffff', linewidths = 2.5, zorder = 3) # plotting contour in white, to make it stand out
                ax.contour(x, y, kde[:,:,i], [0.25], colors = col[i], linewidths = 1.75, zorder = 3) # plotting contour in colour
                ax.scatter(cx, cy, s = 25, c = col[i], edgecolor = '#ffffff', lw = 0.375, zorder = 4) # plotting cluster centroid

            if c != 0: # removing y axes from all tiles EXCEPT those on left-most column
                ax.axes.get_yaxis().set_visible(False)
            if r != d-1: # removing x axes from all tiles EXCEPT those on bottom row
                ax.axes.get_xaxis().set_visible(False)

        if c == r: # setting up tiles for histograms
            ax = MP.subplot2grid((d,d), (r,c)) # setting up axes
            ax.set(xlabel = feat_lbl[c], xlim = (feat_lim[c][0], feat_lim[c][1]), xticks = feat_tix[c])
            for i in range(k): # plotting histograms
                ax.hist(data[clst[:,0] == i][:,c], bins = 'fd', histtype = 'step', color = '#ffffff', lw = 2.0) # in white, to make histogram stand out
                ax.hist(data[clst[:,0] == i][:,c], bins = 'fd', histtype = 'step', color = col[i], lw = 1.25) # in colour
            ax.axes.get_yaxis().set_visible(False) # removing y axes from all tiles
            if r <  d-1: # removing x axes from all tiles EXCEPT the one on the bottom row
                ax.axes.get_xaxis().set_visible(False)

        if c == d-1 and r == 0: # setting up tile for legend
            ax = MP.subplot2grid((d,d), (r,c)) # setting up axes
            ax.legend(handles = leg, loc = 'upper center', fontsize = 10, frameon = True, facecolor = '#ffffff', edgecolor = '#000000', framealpha = 1.0, fancybox = True, borderpad = 0.5) # adding legend
            ax.axes.get_yaxis().set_visible(False) # removing y axis
            ax.axes.get_xaxis().set_visible(False) # removing x axis
            ax.axis('off') # removing frame

#########
# lines #
#########
            
        if c == 1 and r > 1: # adding a line to tiles in second col, and third row and up
            ax.plot([1.9, 1.9], [-20, 20], c = '#000000', ls = '--', zorder = 5)

# lines have to be manually added unfortunately
# use the if statement to specify which tiles you want the line in
# columns start with the first feature (data[:,0]), rows start with the second (data[:,1])
            
fig.subplots_adjust(wspace = 0.05, hspace = 0.05) # spacing tiles
fig.savefig('pair_plot.png')
