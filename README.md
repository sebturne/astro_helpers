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

`cluster_plane_corner_plot.py` is a script to plot all of the 2D projections of clustering outcomes determined in a 5D dataset. This is called a "corner plot", or an "everything vs. everything" plot.

Clusters are plotted in each panel as a single coloured contour at 33% density, determined using a grid search for a best-fitting kernel density estimate in that panel.

Plots/mstar_smdust_ssfr_hlr_bt_tln_corner_k7.png is an example of a plot created using this script.

___

`cluster_feat_imp.py` determines the importance of each of a series of features to a clustering outcome via mutual information.

Plots/feat_imp.png is an example of a plot created using this script (featuring two datasets instead of just one).