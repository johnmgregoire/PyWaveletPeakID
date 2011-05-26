import PyWaveletPeakID as pywaves
import pylab, numpy, time

###CODE BLOCK 1: if you have not downloaded the .npz database of wavelets, run the following line once. Make sure WAVELETDATABASEPATH is set appropriately
#pywaves.createwaveletdatabase()

###CODE BLOCK 2: just read and plot some wavelets at 4 different scales. 
###For each scale, a waveset containing wavelets at all relevant positions will be plotted 
### a waveset contains wavelets with all possible distances from the dataset edges.  
### The central wavelet in the following plots will have "perfect energy" because it does not 
### intercept the edge of the dataset. The other wavelets will be corrected more 
### the close they are to the edge of the dataset until the amount of
### correction,  indicated by the wavelet energy passes the threshold,  i.e.
### wavelets too close to the egde are omitted.
#waves, scales=pywaves.getwavesetfromdatabase([2., 33., 70., 130.]) # the scales must be a subset of those in the database, otherwise the closest available scale will be used
#print scales
#for i, (s, w) in enumerate(zip(scales, waves)):
#    print 'waveset scale:', s,' waveset shape: ', w.shape
#    numnan=numpy.isnan(w[:, 0]).sum()
#    print "number of wavelet whose energy is too far off: ", numnan
#    pylab.subplot(2, 2, i+1)
#    for a in w:
#        if numpy.isnan(a[0]):
#            continue
#        pylab.plot(a)
#    pylab.title('waveset for scale %.1f\n%d good wavelets and %d with bad energy' %(s, w.shape[0]-numnan, numnan))
#print 'check the minimum energy of each wavset:', [min([(ww**2).sum() for ww in w if not numpy.isnan(ww[0])]) for w in waves]
#pylab.show()



###CODE BLOCK 3: create seom fake data and perform peak ID and then plot
###data must be a 2-d array where the first dimension corresponds to independent spectra and the second is the x-axis
#data=numpy.float32([numpy.exp(-1.*((10.-numpy.linspace(0., 20., 2000))/.5)**2)])
#data[0]+=.2*numpy.exp(-1.*((11.-numpy.linspace(0., 20., 2000))/.5)**2)
#print 'data shape:',  data.shape
#
#sg=pywaves.scalegrid_scalesubset(stopscale=80.,startscale=2.,scalemultiplier=1.5)
#scalevals=pywaves.scale_scalegrid_ind(sg)
#print scalevals
#t=time.time()
#waves, sc=pywaves.getwavesetfromdatabase(scalevals)
#print time.time()-t,  'to make waveset'
#
#t=time.time()
#
#wtset=pywaves.wavetrans1d(data, waves, scalevals=scalevals)
#print time.time()-t,  'to get wt'
#
#t=time.time()
#ridgesset=pywaves.ridges_wavetrans1d(wtset, scalevals, noiselevel=0., numscalesskippedinaridge=1.5, padfirstdimintoarray=False, verbose=True)
#print len(ridgesset), [r.shape for r in ridgesset]
#peaks=pywaves.peaks_ridges1d(wtset, ridgesset, scalevals, padfirstdimintoarray=False, minridgelength=3, minchildlength=0., maxscale_localmax=None, minridgewtsum=0., minchildwtsum=0.)
#print len(peaks), [r.shape for r in peaks]
#print time.time()-t,  'to get ridges and peaks'
#
#fig=pylab.gcf()
#wpc=pywaves.wavelet1dplotclass(fig, sg, numposns=len(data[0]))
#wpc.display_wavetrans1d(wtset[0], ridgesset[0], data[0], datascaleind=None, datapeakind=peaks[0][1])
#pylab.show()



print 'done'
