import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot
import matplotlib.offsetbox
import matplotlib._png

def PH5(hubb):
    """
    Recodes GAMA survey's Hubble-like morphological galaxy classes as custom classes for 
    easier plotting.
    With finer classes than PH2().
    """
    plothubb = numpy.zeros((numpy.shape(hubb)[0],1))

    oldhubb = [1,2,11,12,13,14,15] # GAMA values
    newhubb = [1,5, 2, 2, 3, 3, 4] # 'plot friendly' values

    for i in range(len(oldhubb)): # re-assigning Hubble types
        plothubb[:,0][hubb[:,1] == oldhubb[i]] = newhubb[i]

    return plothubb

def PH2(hubb):
    """
    Recodes GAMA survey's Hubble-like morphological galaxy classes as custom classes for 
    easier plotting.
    With coarser classes than PH5().
    """
    plothubb = numpy.zeros((numpy.shape(hubb)[0],1))

    oldhubb = [1,2,11,12,13,14,15] # GAMA values
    newhubb = [1,2, 1, 1, 2, 2, 2] # 'plot friendly' values

    for i in range(len(oldhubb)): # re-assigning Hubble types
        plothubb[:,0][hubb[:,1] == oldhubb[i]] = newhubb[i]

    return plothubb

def FS(feat, clst):
    """
    Determines a sort order for cluster labels (clst) based on a particular feature (feat).
    """
    c = int(max(clst)) + 1
    ordr = range(1,c)
    mean = [0] * len(ordr)
    for i in range(len(ordr)):
        mean[i] = numpy.mean(feat[clst == i+1])
    return [o for (f,o) in sorted(zip(mean,ordr))]

def PS(im, w, coords):
    """
    Crops and neatly arranges galaxy images on a plot canvas. Used to present images of 
    galaxies constituting clusters.
    """
    pics = [0] * len(im)
    boxs = [0] * len(im)
    
    rows = len(im) // w
    grid = numpy.zeros((rows,w), dtype = N.int)

    for r in range(rows):
        grid[r,:] = range(r,len(im),rows)

    for q in range(len(pics)):
        pics[q] = matplotlib._png.read_png('../../Data/png/G' + str(int(im[q])) + '.png')
        size = numpy.shape(pics[q])[0]
        crop = size // 4
        pics[q] = pics[q][crop:size-crop,crop:size-crop]
        boxs[q] = matplotlib.offsetbox.OffsetImage(pics[q], zoom = 0.27)
    
    hpk = [0] * rows

    for r in range(rows):
        hpk[r] = matplotlib.offsetbox.HPacker(children = [boxs[i] for i in grid[r,:]], pad = 0, sep = 0)

    vpk = matplotlib.offsetbox.VPacker(children = [hpk[r] for r in range(rows)], pad = 0, sep = 0)

    pos = coords
    
    return matplotlib.offsetbox.AnnotationBbox(vpk, pos, pad = 0, frameon = False)

def GS(vecs):
    """
    Gram-Schmidt orthonormalisation of input vectors.
    """
    basis = []
    for v in range(vecs.shape[0]):
        w = v - numpy.sum(N.dot(v,b)*b for b in basis)
        if (w > 1e-10).any():  
            basis.append(w / numpy.linalg.norm(w))
    return numpy.array(basis)

def GSC(X):
    """
    QR decomposition.
    """
    Q, R = numpy.linalg.qr(X)
    return Q

def SFH_PLOT(i, s, d, p, z, m):
    """
    Plots star formation histories of galaxies. 
    Intended for use with EAGLE simulation outputs.
    i - galaxy ID.
    s - star formation rate for each galaxy at each EAGLE snapshot.
    d - present day star formation rate for each galaxy (to contextualise history).
    p - subdirectory tag.
    z - redshift of each EAGLE snapshot.
    m - merger mass ratio for each galaxy at each EAGLE snapshot (to distinguish between 
        internally and externally driven growth).
    """
    matplotlib.pyplot.rcParams.update({'figure.figsize':[4,4], 'figure.dpi': 300, 'font.family': 'Times New Roman', 'font.size': 10, 'text.usetex': True})

    d = str(d)
    
    fig, ax = matplotlib.pyplot.subplots()
    ax.plot(N.linspace(0, 8, num = 41)[:-1], s, color = '#000000')
    ax.plot([0, 8], [-11, -11], ls = '--', color = '#000000')
    ax.plot([1, 1], [-14, -8], ls = '--', color = '#000000')

    for q in range(len(z)):
        if m[q] > 0.25:
            ax.plot([z[q], z[q]], [-14, -8], c = 'r')
        else:
            ax.plot([z[q], z[q]], [-14, -8], c = 'orange')
    
    ax.set_title(d)
    ax.set(xlim = (0, 8), xlabel = 'Lookback time (Gyr)', ylim = (-13.1, -8.25), ylabel = r'$sSFR$ (Gyr$^{-1}$)')
    fig.savefig('Post/SFHs/' + p + '/' + str(int(i)) + '.png', bbox_inches = 'tight')
    matplotlib.pyplot.close(fig)

def H(p, q):
    """
    Computes Hellinger distance (measure of similarity between two distributions).
    """
    return numpy.sqrt(numpy.sum((numpy.sqrt(p) - numpy.sqrt(q)) ** 2)) / (numpy.sqrt(2))

def MRPH_PLOT(i, e, p, z, m):
    """
    Plots morphology histories of galaxies.
    Intended for use with EAGLE simulation outputs.
    i - galaxy ID.
    e - morpohology for each galaxy at each EAGLE snapshot.
    p - subdirectory tag.
    z - redshift of each EAGLE snapshot.
    m - merger mass ratio for each galaxy at each EAGLE snapshot (to distinguish between 
        internally and externally driven growth).
    """
    matplotlib.pyplot.rcParams.update({'figure.figsize':[4,4], 'figure.dpi': 300, 'font.family': 'Times New Roman', 'font.size': 10, 'text.usetex': True})

    l = [0., 1.35, 2.32, 3.24, 4.12, 5.22, 5.98, 6.69, 7.35, 7.96]
    
    fig, ax = matplotlib.pyplot.subplots()
    ax.plot(l, e[:len(l)], color = '#000000')

    for q in range(len(z)):
        if m[q] > 0.25:
            ax.plot([z[q], z[q]], [0,1.3], c = 'r')
        else:
            ax.plot([z[q], z[q]], [0,1.3], c = 'orange')

    ax.set(xlim = (0, 8), xlabel = 'Lookback time (Gyr)', ylim = (0, 1.3), ylabel = '$B/T_{m}$')
    fig.savefig('Post/Morphs/' + p + '/' + str(int(i)) + '.png', bbox_inches = 'tight')
    matplotib.pyplot.close(fig)
