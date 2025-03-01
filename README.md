# SPDCTools
Tools used to calculate spontaneous parametric down-conversion processes in nonlinear optical materials.

This repository contains python code files used to calculate the spectra and emission probability of photon pairs as discussed in the paper "A simple model for entangled photon generation in resonant structures".

All code uses either commonly maintained tools or packages, or the custom packages are self-contained in the repository. 

If you run into any issues, please contact the author. 


BEFORE USING THE CODE:

1. Run PerformLithiumNiobateCalcs.py to create several plots on the production of photon pairs. The code can be adapted to consider different crystals, materials, or device geometries. 

2. A custom plot function (plot_custom.py) is used to produce many of the plots. It requires a local installation of latex to function correctly. If you cannot get this working, it is sufficient to change the plotting functions to your liking. 

3. The code uses refractive indices measured and published in several different manuscripts. The sources of the data are contained in the paper titled "A simple model for entangled photon generation in resonant structures" by Nicholas J. Sorensen et al. Please reference that paper for more information on the calculations and the data sources. The sources of data include but are not limited to:

- Green, M. A. (2008). Self-consistent optical parameters of intrinsic silicon at 300 K including temperature coefficients. Solar Energy Materials and Solar Cells, 92(11), 1305–1310. https://doi.org/10.1016/J.SOLMAT.2008.06.009

- Small, D. L., Zelmon, D. E., & Jundt, D. (1997). Infrared corrected Sellmeier coefficients for congruently grown lithium niobate and 5 mol. % magnesium oxide–doped lithium niobate. JOSA B, Vol. 14, Issue 12, Pp. 3319-3322, 14(12), 3319–3322. https://doi.org/10.1364/JOSAB.14.003319


