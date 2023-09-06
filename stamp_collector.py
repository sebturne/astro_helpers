import astro_helpers
import numpy
import requests
import sklearn.neighbors

feat_set = 'mstar_smdust_ssfr_hlr_bt' # features
k = 7 # number of clusters

data = numpy.genfromtxt('Pre/' + feat_set + '_tl.txt', delimiter = ',') # feature data
idsz = numpy.genfromtxt('Pre/id_' + feat_set + '_tln.txt', delimiter = ',') # galaxy ids
imgs = numpy.genfromtxt('Pre/id_img_tln.txt', dtype = 'str', delimiter = ',')[:,1] # galaxy image urls

lbls = numpy.genfromtxt('../../Data/Clustering/EAGLE/RefL0100N1504_z_0/Post/Labels/' + feat_set + '_tln_lbls_k%i.txt' % k, delimiter = ',')
phi  = numpy.genfromtxt('Post/Clustering/' + feat_set + '_tln_staco.txt', delimiter = ',')[:,1:]
phi  = phi[phi[:,1] == k,:]

best = numpy.where(phi[:,0] == min(phi[:,0]))[0][0]
clst = lbls[:,best].reshape(-1,1) # best-fitting labels

cnm = ['E' + str(i) for i in range (7,0,-1)][::-1] # cluster names

cntr = numpy.zeros((k,data.shape[1]))

for i in range(k): # calculating cluster means in all five features
    cntr[i,:] = numpy.mean(data[clst[:,0] == cod[i],:], axis = 0)
    
# filtering out galaxies with no images
data = data[imgs != 'No']
idsz = idsz[imgs != 'No']
clst = clst[imgs != 'No']
imgs = imgs[imgs != 'No']

kdtr = sklearn.neighbours.KDTree(data) # determine kd tree in feature space

for i in range(k):
    ccntr = cntr[i,:].reshape(1,-1)
    cinds = numpy.transpose(kdtr.query(ccntr, k = 64)[1]) # get 64 nearest galaxies to each cluster mean

    cidsz = idsz[cinds] # get their ids
    cimgs = imgs[cinds] # get their image urls

    for j in range(cidsz.shape[0]): # download and save images
        with open('Post/Stamps/' + cnm[i] + '/' + str(int(cidsz[j,0])) + '.png', 'wb') as handle:
            print(cimgs[j,0][10:-2])
            response = requests.get(cimgs[j,0][10:-2], stream = True)

            if not response.ok:
                print(response)

            for block in response.iter_content(1024):
                if not block:
                    break

                handle.write(block)
