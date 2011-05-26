John M Gregoire 26 May 2011

All code written by John Gregoire with adpations of code snippets where indicated. Concepts and code developed through collaboration with Professor R. Bruce van Dover and Darren Dale.

This is a Python implementation of the algorithms published in the following manusciprt:
John M. Gregoire, Darren Dale, and R. Bruce van Dover
A wavelet transform algorithm for peak detection and application to powder x-ray diffraction data
Rev. Sci. Instrum. 82, 015105 (2011)

The manuscript's permalink is
http://link.aip.org/link/doi/10.1063/1.3505103

Written in Python 2.6 and uses numpy and matplotlib packages. The illustrative file example.py uses some additioal packages.

Run and inspect example.py to get a handle on how to use the functions in PyWaveletPeakID.py - there are 3 different code blocks that can be run. 

The intent is to use LoG wavelets to find peaks in a set of 1-d spectra. The units of the spectra are ignored, i.e. the arrays corresponding to x-axis have units of indeces.

To make calculations efficient, a database of wavelets can be caclulated and saved as a .npz. This can take tens of minutes to run but afterwards your analysis will take ~1 second.
BEFORE RUNNING ANYTHING, CHANGE THE PATH CONSTANT IN PyWaveletPeakID.py "WAVELETDATABASEPATH" to the location on your local drive where you want to store the database.
The createwaveletdatabase function uses constants PyWaveletPeakID.py to determine which wavelets to clcualte and save. The default values are meant to create a database that is broadly applicable. One tripping point could be if the data is "oversampled", i.e. if a peak in the data is composed of a million data points, you may want to bin the data before performing wavelet transforms. If these are changed, the database must be recalculated before performing analysis.

PARAMETERS TO CHANGE THE BEHAVIOR, SENSIIVTY, ETC. OF THE PEAK IDENTIFICATION - if you read the paper you will learn a lot more about these parameters
. maxfixenfrac: specifies the range of relative wavelet energy to use in the calculation. wavelets near the edge of the dataset or with very few points per scale will have an 'incorrect' energy and this can be fixed, but fixing wavelets that are way off can introduce artifacts, so this is a balance between having a more complete wavelet transform and introducing artifacts into the wavelet transform
. scaleweightingfcn_inverse: this is the function of the wavelet scale by which you wieght the wavelet transform. For my purposes, 1/scale has always worked.
. noiselevel: value of wavelet transform under shich peaks will be ignored
. numscalesskippedinaridge: ridges are connections among scales, values of this parameter >1 will allow "holes" in the ridge
. critsepfactor: the critical separation of wavelet transform peaks to make a mother-child connection is critsepfactor*<wavelet scale>. Optimal values for this parameter can be calculated for a given peak shape in your data, but imperical determination of this parameter is easy - if you are seeing too many instances of ridges being split into children, reduce this parameter. If you are missing split peaks and especially if you see a single peak in the data whose ridge is being split into children, increase this parameter.
. minridgelength: minimum length (i.e. number of wavelet scales) of ridge for it to count as a peak
. minchildlength: same thing but for length of the child ridge alone, not count its mother's length
. maxscale_localmax: the largest scale in which the ridge can be locally maximum and still count as a peak
. ***minridgewtsum: critical value for the sum of the wavelet transform over the ridge. This will depend on the number of wavelet scales used and the nominal intensities of the original data.
. ***minchildwtsum: same thing except for the child ridge. The child ridge must have this total intensity to count as a peak, but for the "minridgewtsum", the child and mother intensities are added together.
***MOST IMPORTANT

known issues and potential pitfalls:

. docstrings are only provided for a few functions and even these are incomplete

. A lot of the math is done with data indeces represented as int16 and the "data index" value 32767 is reserved to mean "no index" or the equivalent of a float NaN. If the x-axis of your data has more than 32766 values, the int16 will need to be replaced with a higher bit integer and the 32767 replace with corresponding max-value.

. the plotting routine attempts to relabel the axes ticks but there is a bug somewhere

. The ideal values for the plotting parameters such as linewidth will vary with the user's data size and number of scales but there is no user interface for adjusting these.