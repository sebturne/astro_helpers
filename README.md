# astro_highlights
Just a small collection of scripts and functions I wrote during my time as an astrophysicist. See my stacopy repo for further astro work.

___

`astro_helpers.py` contains a series of helper functions I wrote to conduct my analyses.

Plots/E5_stamp.png is an example of a plot that was made using `PS()`.

Further example plots pending...

___

`hlr_snippet.py` is a pedagogical script I wrote to demonstrate the calculation of various 
cosmological distance measures from GAMA survey redshifts in a 737 cosmology.

___

`cluster_plane_single_plot.py` is a script to plot a 2D projection of clustering outcomes determined in a 5D dataset.

Clusters are plotted as a single coloured contour at 33% density, determined using a grid search for a best-fitting kernel density estimate.

Plots/mstar_smdust_ssfr_hlr_bt_tln_k7_04.png is an example of a plot created using this script.

___

`cluster_pair_plot.py` is a pedagogical script I wrote to explain how to make a "pair plot" (AKA "corner plot" or "everything vs. everything plot"). The plot projects an input dataset and clusters determined within it onto every possible features. The intent is to identify projections within which clusters are well-separated and, consequentially, features that were important to the clustering. 

Clusters are plotted in each panel as a single coloured contour, determined using a grid search for a best-fitting kernel density estimate in that panel.

Plots/mstar_smdust_ssfr_hlr_bt_tln_corner_k7.png is an example of a plot created using a script like this.

___

`cluster_feat_imp.py` determines the importance of each of a series of features to a clustering outcome via mutual information.

Plots/feat_imp.png is an example of a plot created using this script (featuring two datasets instead of just one).

___

`stamp_collector.py` collects images ("postage stamps" because they're small and square) of galaxies. It gets images of the 64 most representative galaxies in each cluster. It does this by conducting a k-nearest neighbours search in the 5D input feature space. It then goes and gets the images from the EAGLE public database, where they're stored.

___

`hierarchy_plot.py` is a script that generates a (very rudimentary!) Sankey diagram from scratch, to compare relationships between clustering outcomes consisting of different numbers of clusters. Description TBC...