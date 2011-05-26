import numpy
import numpy.ma as ma
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.mlab as mlab

WAVELETDATABASEPATH='C:/Users/JohnnyG/Documents/PythonCode/waveletpackage/waveletdatabase.npz'
DATABASE_NUMSCALES=50
DATABASE_NSCALES_SIDE=4
DATABASE_STARTSCALE=6.
DATABASE_SCALEMULTIPLIER=10.**(numpy.log10(2.)/10.)

def createwaveletdatabase(numscales=DATABASE_NUMSCALES, nscales_side=DATABASE_NSCALES_SIDE, startscale=DATABASE_STARTSCALE, scalemultiplier=DATABASE_SCALEMULTIPLIER, savepath=WAVELETDATABASEPATH):
    """
    createwaveletdatabase(<DEFAULTS SET AS CONSTANTS IN PyWaveletsPeakID>)
    This will create a database of wavelets with differenet scales, where "scale" is a parameter in the LoG function.
    Everything i in units of indeces so the smallest scale will be "startscale" datapoints and the scales will be
    log-spaced by factors of "scalemultiplier", with a total number of scales "numscales".
    Each wavelet has finite value out to infinity but the practical window size of a wavelet is considered to be +- 4 or 5 wavelet scales so the 
    wavelets will only be calculated with nscales_side*scale data points on each side of the wavelet center.
    """
    scales=numpy.float32(startscale*scalemultiplier**numpy.array(range(numscales)))
    fixenwaves=[buildwaveset1d([s, 0., 1], [0., 1., int(nscales_side*s)+1], [0., 1., 2*int(nscales_side*s)+1], maxfixenfrac=9999999.) for s in scales]
    fixen=[]
    waves=[]
    for fe, wv in fixenwaves:
        fixen+=[fe[0]]
        waves+=[wv[0].flatten()]

    fixen=numpy.float32(numpy.concatenate(fixen))
    waves=numpy.float32(numpy.concatenate(waves))
    #print [[fe.min(), fe.max()] for fe in fixen]
    numpy.savez(savepath, scales=scales, nscales_side=numpy.uint32([nscales_side]), fixen=fixen, waves=waves)
    
    
def scalegrid_scalesubset(stopscale=32., startscale=6., scalemultiplier=1.1, masternumscales=DATABASE_NUMSCALES, masternscales_side=DATABASE_NSCALES_SIDE, masterstartscale=DATABASE_STARTSCALE, masterscalemultiplier=DATABASE_SCALEMULTIPLIER):
    m=masterscalemultiplier**int(numpy.log(scalemultiplier)/numpy.log(masterscalemultiplier))
    n=numpy.round(numpy.log(stopscale/startscale)/numpy.log(m))+1
    return [startscale, m, n]
    
def savgolsmooth(x, nptsoneside=7, order = 4, dx=1.0, deriv=0): #based on scipy cookbook. x is 1-d array, window is the number of points used to smooth the data, order is the order of the smoothing polynomial, will return the smoothed "deriv"th derivative of x
    side=numpy.uint32(max(nptsoneside, numpy.ceil(order/2.)))
    s=numpy.r_[2*x[0]-x[side:0:-1],x,2*x[-1]-x[-2:-1*side-2:-1]]
    # a second order polynomal has 3 coefficients
    b = numpy.mat([[k**i for i in range(order+1)] for k in range(-1*side, side+1)])
    m = numpy.linalg.pinv(b).A[deriv] #this gives the dth ? of the base array (.A) of the pseudoinverse of b

    # precompute the offset values for better performance
    offsets = range(-1*side, side+1)
    offset_data = zip(offsets, m)

    smooth_data=[numpy.array([(weight * s[i + offset]) for offset, weight in offset_data]).sum() for i in xrange(side, len(s) - side)]
    smooth_data=numpy.array(smooth_data)/(dx**deriv)
    return smooth_data
    
def firstder(y, dx=1.0):
    """firstder(y, dx=1.0)
    takes an array of values y and the dx corresponding to 1 index and 
    returns same length array of the 5-point stencil of second derivative, 
    copying the 2 values on each end.
    """
    secd=(8.0*(y[3:-1]-y[1:-3])+y[:-4]-y[4:])/(12.0*dx)
    temp=numpy.empty(y.shape, dtype=numpy.float32)
    temp[:2]=secd[0]
    temp[-2:]=secd[-1]
    temp[2:-2]=secd[:]
    return temp

def secder(y, dx=1.0):
    """secder(y, dx=1.0)
    takes an array of values y and the dx corresponding to 1 index 
    and returns same length array of the 5-point stencil of second 
    derivative, copying the 2 values on each end
    """
    secd=((-30.0)*y[2:-2]+16.0*(y[1:-3]+y[3:-1])-(y[:-4]+y[4:]))/(12.0*dx**2)
    temp=numpy.empty(y.shape, dtype=numpy.float32)
    temp[:2]=secd[0]
    temp[-2:]=secd[-1]
    temp[2:-2]=secd[:]
    return temp
    
def arrayzeroind1d(arr, postoneg=False):
    sarr=numpy.sign(arr)
    if postoneg:
        zeroind=numpy.where(sarr[:-1]>sarr[1:])[0]
    else:
        zeroind=numpy.where(sarr[:-1]*sarr[1:]<=0)[0]
    return (1.0*zeroind*arr[(zeroind+1,)]-(zeroind+1)*arr[(zeroind,)])/(arr[(zeroind+1,)]-arr[(zeroind,)]) #returns array of the floating point "index" linear interpolation between 2 indeces

def clustercoords1d(pkind, critqsepind):#results will be sorted
    pkind.sort()
    newpks=[]
    i=0
    while i <(len(pkind)-1):
        if (pkind[i+1]-pkind[i])<critqsepind:
            newpks+=[(pkind[i+1]+pkind[i])/2.0]
            i+=2
        else:
            newpks+=[pkind[i]]
            i+=1
    return newpks #not exactly centroid but close enough

def clustercoordsbymax1d(arr, pkind, critqsepind):
    """
    clustercoordsbymax1d(arr, pkind, critqsepind)
    results will be sorted. wherever there are peak indeces too close together. 
    the peak index next to the peak index with highest arr value gets removed
    """
    pkind.sort()
    indindslow=numpy.where((pkind[1:]-pkind[:-1])<critqsepind)[0]
    indindshigh=indindslow+1
    while indindslow.size>0:
        maxindindindlow=numpy.nanargmax(arr[pkind[(indindslow,)]])
        maxindindindhigh=numpy.nanargmax(arr[pkind[(indindshigh,)]])
        if arr[pkind[indindslow[maxindindindlow]]]>arr[pkind[indindshigh[maxindindindhigh]]]:
            pkind=numpy.delete(pkind, indindshigh[maxindindindlow])
        else:
            pkind=numpy.delete(pkind, indindslow[maxindindindhigh])

        indindslow=numpy.where((pkind[1:]-pkind[:-1])<critqsepind)[0]
        indindshigh=indindslow+1
    return pkind
    
def peaksearch1d(innn, dx=1., critcounts=10, critqsepind=5, critcurve=None, max_withincritsep=False): #dx is delta q for one index. zeros of the first derivative of inn are grouped together if within critsepind. only negative slope in the firstder is used so no secder is necessary unless specify a critical curvature in count nm^2
    ifirstder=firstder(innn, dx)
    zeroind=arrayzeroind1d(ifirstder, postoneg=True)
    temp=numpy.where(innn[(numpy.uint32(numpy.round(zeroind)),)]>critcounts)
    fullpkind=zeroind[temp]
    if fullpkind.size==0:
        return fullpkind
    if max_withincritsep:
        pkind=clustercoordsbymax1d(innn, numpy.uint32(numpy.round(fullpkind)), critqsepind)
    else:
        pkind=clustercoords1d(fullpkind, critqsepind)#these pk indeces are floating point!!!
    if critcurve is not None:
        isecder=secder(innn, dx)
        temp=numpy.where(isecder[(numpy.uint32(numpy.round(pkind)),)]<(-1*critcurve))
        pkind=numpy.array(pkind)[temp]
    pkind=list(pkind)
    pkind.reverse()#highest to smallest for pairing below
    return numpy.array(pkind, dtype=numpy.float32)
    
def buildwaveset1d(qscalegrid, qposngrid, qgrid, maxfixenfrac=0.12, enfractol=0.0, maxoverenergy=None):
    ENERGY=0.57457 #this is constant fro all scales and translations
    maxfixenfrac+=1 #this notes a discrepancy between fixenfrac in GUI and in code and saved attribute
    if maxoverenergy is None:
        maxoverenergy=maxfixenfrac
    waveattrdict={'qgrid':qgrid, 'qscalegrid':qscalegrid, 'qposngrid':qposngrid,'ENERGY':ENERGY, 'maxfixenfrac':maxfixenfrac, 'maxoverenergy':maxoverenergy, 'enfractol':enfractol}

    dq=qgrid[1]
    waveset=waveletset1d(qgrid, qscalegrid, qposngrid)
    a, b, c=waveset.shape
    fixenarr=numpy.empty((a, b), dtype='float32')
    for i in range(a):
        for j in range(b):
            en=((waveset[i, j, :]**2)*dq).sum()
            fixenfrac=ENERGY/en
            if fixenfrac<maxfixenfrac and 1/fixenfrac<maxoverenergy:
                if en<((1.0-enfractol)*ENERGY) or en>((1.0+enfractol)*ENERGY):
                    waveset[i, j, :]=wave1dkillfix(waveset[i, j, :], ENERGY, dq=dq)*dq
                else:
                    fixenfrac=0.0
                    waveset[i, j, :]*=dq
                fixenarr[i, j]=fixenfrac
            else:
                waveset[i, j, :]*=numpy.nan
                fixenarr[i, j]=numpy.nan

    if numpy.isnan(fixenarr).sum()==fixenarr.size:
        print 'every wavelet calculation resulted in error. check energy. nothing saved'
        return

    
    return fixenarr, waveset


def scaleweightingfcn_inverse(sc):
    return 1./sc
def scaleweightingfcn_inverse3(sc):
    return 1./sc**3
    
def wavetrans1d(data, waveset, scalevals=None, scaleweightingfcn=scaleweightingfcn_inverse):
    """
    wavetrans1d(data, waveset, scalevals=None, scaleweightingfcn=scaleweightingfcn_inverse)
    data must be a 2-d array where the first dimension corresponds to independent spectra and the second is the x-axis. i.e. data is nsets x npts and npts must be larger than the length of the wavelets
    waveset is a list over scales with each element a 2-d square array of waveposns x dataposns.    
    scaleweightingfcn_inverse: this is the function of the wavelet scale by which you wieght the wavelet transform and if used, scalevals must be passed
    """

    n=data.shape[1]
    starti=[[0]*(w.shape[0]//2)+range(n-w.shape[0]+1)+[n-w.shape[0]]*(w.shape[0]//2) for w in waveset]
    wavi=[range(w.shape[0]//2)+[w.shape[0]//2]*(n-w.shape[0]+1)+range(w.shape[0]//2+1, w.shape[0]) for w in waveset]
    if scalevals is None or scaleweightingfcn is None:
        wghts=numpy.ones(len(waveset), dtype='float32')
    else:
        wghts=scaleweightingfcn(scalevals)
        #                                                                                                                                                          one index for every posn              one set of indeces, and waves(posn) for each scale                                      one 1d data array for every member of the dataset
    return numpy.float32([[[(d[i:i+waves.shape[0]]*waves[j]).sum()*wght for i, j in zip(si, wi)] for si, wi, waves, wght in zip(starti, wavi, waveset, wghts)] for d in data])


def invwavetrans1d(wavetrset, waveset, scalevals=None, scaleweightingfcn=scaleweightingfcn_inverse3):
    """
    THIS FUNCTION DOESN'T WORK YET. SOME EFFORT IS NEEDED TO DEAL WITH THE SCALE SPACING AND OTHER WEIGHTINGS
    waveset is a list over scales with each element a 2-d square array of waveposns x dataposns.    
    data is nsets x npts and npts must be larger than the wavelets
    """
#    if data.ndim==1:
#        data=[data]
#        returnfirst=True
#    else:
#        returnfirst=False
    n=wavetrset.shape[2]
    starti=[[0]*(w.shape[0]//2)+range(n-w.shape[0]+1)+[n-w.shape[0]]*(w.shape[0]//2) for w in waveset]
    wavi=[range(w.shape[0]//2)+[w.shape[0]//2]*(n-w.shape[0]+1)+range(w.shape[0]//2+1, w.shape[0]) for w in waveset]
    if scalevals is None or scaleweightingfcn is None:
        wghts=numpy.ones(len(waveset), dtype='float32')
    else:
        wghts=scaleweightingfcn(scalevals)
#    try:
#        numpy.float32([numpy.float32([[(d[i:i+waves.shape[0]]*waves[j]).sum()/wght for i, j in zip(si, wi)] for si, wi, waves, wght, d in zip(starti, wavi, waveset, wghts, wt)]).sum(axis=0) for wt in wavetrset])
#    except:
#        print wavetrset.shape, waves.shape, wt.shape, d.shape, d[i:i+waves.shape[0]].shape, wght
    dataarrays_scales_posns=numpy.float32([[[(d[i:i+waves.shape[0]]*waves[j]).sum()/wght for i, j in zip(si, wi)] for si, wi, waves, wght, d in zip(starti, wavi, waveset, wghts, wt)] for wt in wavetrset])
    dataarrays_scales_posns[numpy.isnan(dataarrays_scales_posns)]=0.
    return dataarrays_scales_posns.sum(axis=1)
    


def peakfit1d(h5path, h5groupstr, windowextend_hwhm=3, peakshape='Gaussian', critresidual=.2, use_added_peaks=False, type='h5mar'):
    """
    this function is copied from the vanDover_CHESS git repository and has not yet been implemented into this package
    """
    try:
        peakfcn=eval(peakshape)
    except:
        print 'ABORTED: did not understand peak shape "',peakshape,'" - this must be an already defined function.'
        return 'ABORTED: did not understand peak shape "',peakshape,'" - this must be an already defined function.'

    h5file=h5py.File(h5path, mode='r')
    h5analysis=h5file['/'.join((h5groupstr, 'analysis'))]
    h5mar=h5file['/'.join((h5groupstr, 'analysis', getxrdname(h5analysis)))]
    if 'h5mar' in type:
        wtgrpstr='/'.join((h5groupstr, 'analysis', getxrdname(h5analysis), 'wavetrans1d'))
        pointlist=h5analysis.attrs['pointlist']
        ifcountspoint=h5mar['ifcounts']
        numpts=ifcountspoint.shape[0]
        qgrid=h5mar['ifcounts'].attrs['qgrid']
        h5grpstr='/'.join((h5groupstr, 'analysis', getxrdname(h5analysis)))

    elif 'h5tex' in type:
        h5grpname=type.partition(':')[2]
        h5tex=h5mar['texture']
        h5texgrp=h5tex[h5grpname]
        pointlist=h5texgrp.attrs['pointlist']
        wtgrpstr='/'.join((h5groupstr, 'analysis', getxrdname(h5analysis), 'texture', h5grpname, 'wavetrans1d'))
        h5grpstr='/'.join((h5groupstr, 'analysis', getxrdname(h5analysis), 'texture', h5grpname))
        ifcountspoint=h5texgrp['ifcounts']
        numpts=ifcountspoint.shape[0]
        qgrid=h5texgrp['ifcounts'].attrs['chigrid']



    wtgrp=h5file[wtgrpstr]

    qvals=q_qgrid_ind(qgrid)

    qscalegrid=wtgrp.attrs['qscalegrid']
    qposngrid=wtgrp.attrs['qposngrid']

    wtpeakspoint=wtgrp['peaks']

    if 'additionalpeaks' in h5file[h5grpstr] and use_added_peaks:
        addpeaks=readh5pyarray(h5file[h5grpstr]['additionalpeaks'])
    else:
        addpeaks=None

    #pointlist=[41, 49]#***
    qshsss=[] #q values, scale value, height, sigma q scale, sigma sclae, sigma height
    for peakind in pointlist:
        #print 'point', peakind
        counts=ifcountspoint[peakind]
        notnaninds=numpy.where(numpy.logical_not(numpy.isnan(counts)))[0]
        wtpeakdata=wtpeakspoint[peakind, :, :]
        qscales=wtpeakdata[0, :]
        qposns=wtpeakdata[1, :]
        qscales=qscales[qscales!=32767]
        qposns=qposns[qposns!=32767]
        qscales=scale_scalegrid_ind(qscalegrid, qscales)
        #print qscales
        qscales*=0.36 #for wavelet->Gaussian HWHM
        qposns=q_qgrid_ind(qposngrid, qposns)
        if not (addpeaks is None):
            addpeakinds=numpy.where(numpy.uint32(numpy.round(addpeaks[:, 0]))==peakind)
            if len(addpeakinds[0])>0:
                #print addpeaks[addpeakinds, 1], '**',addpeaks[addpeakinds, 2]
                qscales=numpy.append(qscales, addpeaks[addpeakinds, 1])
                qposns=numpy.append(qposns, addpeaks[addpeakinds, 2])
                sortinds=qposns.argsort()
                qposns=qposns[sortinds]
                qscales=qscales[sortinds]
                #print qposns
                #print qscales
        if len(qscales)==0:
            qshsss+=[numpy.float32([[]])]
            continue
        qscales=numpy.float32([max(qs, .25) for qs in qscales])#this is intended for overlapping peaks where wt will give very low qscale.
        indrangeandpeakinds=windows_peakpositions(qgrid, qscales, qposns, windowextend_qscales=windowextend_hwhm)
        #print 'windows', indrangeandpeakinds
        pars=None
        sigs=None
        for indrange, peakinds in indrangeandpeakinds:
            startpars=[[qposns[i], qscales[i], counts[notnaninds[numpy.argmin((notnaninds-ind_qgrid_q(qgrid, qposns[i]))**2)]]] for i in peakinds]
            #print 'startpars',  startpars
            inds=list(set(notnaninds)&set(range(indrange[0], indrange[1])))
            if len(inds)==0:
                print 'THIS WILL CRASH BECUASE THERE ARE NO VALID DATA POINT IN THIS WINDOW - THE DATA INDEX ENDPOINTS BEFORE NANs WERE REMOVED WERE ' , indrange[0], indrange[1]
            p, s, r=fitpeakset(qvals[inds], counts[inds], startpars, peakfcn)
            if pars is None:
                pars=p[:, :]
                sigs=s[:, :]
            else:
                pars=numpy.concatenate((pars,p),axis=0)
                sigs=numpy.concatenate((sigs,s),axis=0)
        qshsss+=[numpy.concatenate((pars.T,sigs.T),axis=0)]
    h5file.close()
    #print qshsss
    #return
    maxnumpeaks=max([arr.shape[1] for arr in qshsss])
    savearr=numpy.ones((numpts, 6, maxnumpeaks), dtype='float32')*numpy.nan
    for pointind, arr in zip(pointlist, qshsss):
        savearr[pointind, :, :arr.shape[1]]=arr[:, :]

    h5file=h5py.File(h5path, mode='r+')
    h5grp=h5file[h5grpstr]
    if 'pkcounts' in h5grp:
        del h5grp['pkcounts']
    pkcounts=h5grp.create_dataset('pkcounts', data=savearr)
    pkcounts.attrs['windowextend_hwhm']=windowextend_hwhm
    pkcounts.attrs['peakshape']=peakshape
    pkcounts.attrs['critresidual']=critresidual

    if not (addpeaks is None):
        h5grp['additionalpeaks'].attrs['usedinfitting']=1
    h5analysis=h5file['/'.join((h5groupstr, 'analysis'))]
    h5file.close()


def getpeaksinrange(h5path, h5groupstr, indlist=None, qmin=0, qmax=1000, returnonlyq=True,  verbose=False, returnonlytallest=True):
    """
    this function is copied from the vanDover_CHESS git repository and has not yet been implemented into this package
    """
    h5file=h5py.File(h5path, mode='r')
    h5analysis=h5file['/'.join((h5groupstr, 'analysis'))]
    h5mar=h5file['/'.join((h5groupstr, 'analysis', getxrdname(h5analysis)))]

    pkcounts=readh5pyarray(h5mar['pkcounts'])
    pointlist=h5analysis.attrs['pointlist']
    returnpointinds=[]
    returnpeakinfo=[]
    if indlist is None:
        indlist=pointlist
    for i in indlist:
        a, b, c, d, e, f=peakinfo_pksavearr(pkcounts[i,:,:], fiterr=True)
        goodinds=numpy.where((a>=qmin)&(a<=qmax))
        if len(goodinds[0])>0:
            if returnonlytallest:
                printindlist=[goodinds[0][numpy.nanargmax(c[goodinds])]]
            else:
                printindlist=goodinds[0]
            for printind in printindlist:
                returnpointinds+=[i]
                if returnonlyq:
                    returnpeakinfo+=[a[printind]]
                    if verbose:
                        print i, '\t', a[printind]
                else:
                    returnpeakinfo+=[[a[printind], b[printind], c[printind], d[printind], e[printind], f[printind]]]
                    if verbose:
                        print '\t'.join((`i`, `a[printind]`, `b[printind]`, `c[printind]`, `d[printind]`, `e[printind]`, `f[printind]`))
            continue
        if verbose:
            print '\t'*6*(1-returnonlyq)
    h5file.close()
    return returnpointinds, numpy.float32(returnpeakinfo) #if indlist had no peaks then it is not in returnointlist

def q_qgrid_ind(qgrid, index='all'):
    if index=='all':
        index=numpy.array(range(numpy.uint32(qgrid[2])), dtype=numpy.float32)
    elif isinstance(index, list):
        index=numpy.array(index)
    return qgrid[0]+qgrid[1]*index

def qgrid_minmaxint(min, max, inter):
    num=(max-min)//inter+1
    return [min, inter, num]

def qgrid_minmaxnum(min, max, num):
    return [min, (1.0*max-min)/(num-1), num]

def minmaxint_qgrid(qgrid):
    return (qgrid[0], qgrid[0]+qgrid[1]*(qgrid[2]-1), qgrid[1])

def slotends_qgrid(qgrid):
    return numpy.array(range(numpy.uint32(qgrid[2])+1), dtype='float32')*qgrid[1]+qgrid[0]-qgrid[1]/2.0

def ind_qgrid_q(qgrid, q, fractional=True):
    if fractional:
        return (1.0*q-qgrid[0])/qgrid[1]
    else:
        return numpy.int32(numpy.round((1.0*q-qgrid[0])/qgrid[1]))

def scale_scalegrid_ind(scalegrid, index='all'):
    if index=='all':
        index=numpy.array(range(numpy.uint32(scalegrid[2])), dtype=numpy.float32)
    elif isinstance(index, list):
        index=numpy.array(index)
    return scalegrid[0]*(scalegrid[1]**index)

def scalegrid_minmaxint(min, max, inter):
    num=int(round(numpy.log(1.0*max/min)/numpy.log(inter)))+1
    return [min, inter, num]

def scalegrid_minmaxnum(min, max, num):
    return [min, numpy.exp(numpy.log(1.0*max/min)/(num-1)), num]

def minmaxint_scalegrid(scalegrid):
    return (scalegrid[0], scalegrid[0]*(scalegrid[1]**(scalegrid[2]-1)), scalegrid[1])

def ind_scalegrid_scale(scalegrid, scale):
    return int(round(numpy.log(1.0*scale/scalegrid[0])/numpy.log(scalegrid[1])))
    
def scaleposngrid_affinegrid(affinegrid):
    scales=scale_scalegrid_ind(affinegrid[:3])
    return [[sc, qgrid_minmaxint(affinegrid[3], affinegrid[4], 1.0*sc/affinegrid[5])] for sc in scales]

def scaleposnlist_affinegrid(affinegrid):
    t=scaleposngrid_affinegrid(affinegrid)
    posnlist=[list(q_qgrid_ind(dup[1])) for dup in t]
    scalelist=[[dup[0]]*len(posns) for dup, posns in zip(t, posnlist)]
    return numpy.array(zip(flatten(scalelist), flatten(posnlist)))
    
def waveletset1d(qgrid, qscalegrid, qposngrid):
#    return numpy.float32([[[1.64795*(1.0-((q-qp)/qs)**2)*numpy.exp(-0.5*((q-qp)/qs)**2)/numpy.sqrt(2.0*numpy.pi*qs) for q in q_qgrid_ind(qgrid)] for qp in q_qgrid_ind(qposngrid)] for qs in scale_scalegrid_ind(qscalegrid)])
    ans=[]
    for qs in scale_scalegrid_ind(qscalegrid):
        ans+=[numpy.float32([[1.64795*(1.0-((q-qp)/qs)**2)*numpy.exp(-0.5*((q-qp)/qs)**2)/numpy.sqrt(2.0*numpy.pi*qs) for q in q_qgrid_ind(qgrid)] for qp in q_qgrid_ind(qposngrid)])]
    return numpy.float32(ans)

def wave1dkillfix(wave, targetenergy, dq=1):
    plist=numpy.where(wave>0)
    nlist=numpy.where(wave<0)
    wavep=wave[plist]
    waven=wave[nlist]
    sump=wavep.sum()
    sumn=-1.0*waven.sum()
    enp=((dq*wavep)**2).sum()
    enn=((dq*waven)**2).sum()
    f=numpy.sqrt(targetenergy/(enp+((sump/sumn)**2)*enn))
    wave[plist]*=f
    wave[nlist]*=(f*sump/sumn)
    return wave


def perform_ridges_wavetrans1d(wtrev, qsindlist, noiselevel, numscalesskippedinaridge=1.5, critsepfactor=3., verbose=False):
    """perform_ridges_wavetrans1d(wtrev, qsindlist, noiselevel, numscalesskippedinaridge=1.5, critsepfactor=3., verbose=False)
    the scale index of the wt has been reversed so that this fucntion steps from biggest to msallest wavelet scale. 
    So the scale index of ridges is inverted from that of the previously saved wt
    """
    initpeakind=list(numpy.int16(numpy.round(peaksearch1d(wtrev[0], dx=1, critcounts=noiselevel, critqsepind=qsindlist[0], max_withincritsep=True))))#this dx no good if using curvature
    ridges=[[ind]+[32767]*(wtrev.shape[0]-1) for ind in initpeakind]
    print 'num init ridges and initpeaks', len(ridges), initpeakind
    for scalecount in range(1, wtrev.shape[0]):
        wtrow=wtrev[scalecount, :]
        peakind=list(numpy.int16(numpy.round(peaksearch1d(wtrow, dx=1, critcounts=noiselevel, critqsepind=qsindlist[scalecount], max_withincritsep=True))))
        for ridgecount, ridge in enumerate(ridges):
            if len(peakind)>0 and ridge[scalecount]==32767: #need peaks to assign and also if ridge forked in previous scale then that ridge has ended
                temp=1
                while ridge[scalecount-temp]==32767 or ridge[scalecount-temp]<0:
                    temp+=1
                if temp-1<=numscalesskippedinaridge:
                    ridgerep=ridge[scalecount-temp]
                    if verbose:
                        print 'PEAKIND, positions of peaks being processed', peakind
                        print 'scale index: ', scalecount
                        print 'RIDGE value, peak posns from high to low scale:', ridge
                        print 'value from RIDGE serving as representative', ridgerep
                        print 'distance**2 of peaks from representative', (1.0*numpy.float32(peakind)-ridgerep)**2
                        print 'critical distance**2 for this scale', (critsepfactor*qsindlist[scalecount-1])**2
                        print ' '
                    
                    closeenoughinds=list(numpy.where((1.0*numpy.float32(peakind)-ridgerep)**2<(critsepfactor*qsindlist[scalecount-1])**2)[0])#a larger critsepfactor loosens the contraint for associating a current peak with one from the previous (larger) qscale and thus makes mother->children associations more common
                    #print 'closeenoughinds1', closeenoughinds
                    allridgereps=numpy.float32([r[scalecount-temp] for r in ridges])
                    closeenoughinds=[ceind for ceind in closeenoughinds if ridgecount==numpy.nanargmin((peakind[ceind]-allridgereps)**2)]#peaks are only close enough to a ridge if that ridge is the closest tot he peak
                    closestind=numpy.nanargmin((numpy.float32(peakind)-ridgerep)**2)
                    if len(closeenoughinds)==1:
                        ridge[scalecount]=peakind.pop(closestind)
                    elif len(closeenoughinds)>1:
                        newridgestart=numpy.int16(ridge[:scalecount])
                        newridgestart[newridgestart!=32767]=-1*ridgecount-1
                        #the child ridge is filled from the largest scale with the ridge index of its mother ridge. NEGATIVE VALUES IN A RIDGE CORRESPOND TO AN INDEX IN THE LIST OF RIDGES
                        newridgestart=list(newridgestart)
                        closeenoughinds.sort(reverse=True) #this is imperative because otherwise the .pop() will mess things up
                        for ceind in closeenoughinds:
                            pkind=peakind.pop(ceind)
                            ridges+=[newridgestart+[pkind]+[32767]*(wtrev.shape[0]-scalecount-1)]
                            if ceind==closestind:
                                ridges[ridgecount]=ridge[:scalecount]+[-1*(len(ridges)-1)-1]*(len(ridge)-scalecount) 
                                #the mother ridge is filled to the lowest scale with the ridge index of its closest child ridge (once a moth has a child ridge it is finished)

        for pkind in peakind:
            ridges+=[[32767]*scalecount+[pkind]+[32767]*(wtrev.shape[0]-scalecount-1)]

    return ridges

def ridges_wavetrans1d(wavetrset, scalevals, noiselevel=None, numscalesskippedinaridge=1.5, padfirstdimintoarray=True, critsepfactor=3., verbose=False):
    """
    ridges_wavetrans1d(wavetrset, scalevals, noiselevel=None, numscalesskippedinaridge=1.5, padfirstdimintoarray=True, critsepfactor=3., verbose=False)
    wavetrset is 3-d, 1stdim: datasets  2nddim: scales in increasing order  3rddim: posns in increasing order
    noiselevel: value of wavelet transform under shich peaks will be ignored
    numscalesskippedinaridge: ridges are connections among scales, values of this parameter >1 will allow "holes" in the ridge
    critsepfactor: the critical separation of wavelet  transform peaks to make a mother-child connection is critsepfactor*<wavelet scale>
    """
    qposnint=1 #in this implementation of wavelets, scalegriwd=posngrid
    qsindlist=[2*max(int(numpy.ceil(1.*qs/qposnint)), 1) for qs in scalevals[::-1]]

    if noiselevel is None:
        noiselevel=numpy.nanmin(wavetrset)
    ridges=[perform_ridges_wavetrans1d(wt[::-1, :], qsindlist, noiselevel, numscalesskippedinaridge=numscalesskippedinaridge, critsepfactor=critsepfactor, verbose=verbose) for wt in wavetrset]
    if padfirstdimintoarray:
        numr=[len(r) for r in ridges]
        maxnr=max(numr)
        filler=[[32767]*wt.shape[0]]*maxnr
        for r in ridges:
            r+=filler[:len(filler)-len(r)]
        return numpy.int16(ridges) #this is 3-d array of numdatasets x maxnumridges x numscales, with 32767 padding in the ridges and scales
    else:
        return [numpy.int16(r) for r in ridges] #this is list of 2-d arrays, one for each dataset,  numridges(varies) x numscales, with 32767 padding in the scales
    
def perform_peaks_ridges1d(wt, ridges, ridgescalecritind=0, minridgelength=1, minchildlength=1, minridgewtsum=0., minchildwtsum=0., verbose=False):#wt scale ind is small->big but ridges is big->small and ridgescalecritind is of ridges
    ridgeinds=numpy.where(((ridges!=32767).sum(axis=1)>=minridgelength)&(ridges[:, -1]!=32767))[0] #this is the ridge length including the "good" ridge components from other ridges associated through forking that has to be at least minridgelength but this is only good if the ridge goes to the smallest scale
    ridgeinds2=numpy.where(((ridges!=32767)*(ridges>=0)).sum(axis=1)>=minridgelength)[0]#this will catch the ridges that don't go to the end but are long enough on their own (not counting mother forks). mother forks ruled out later
    ridgeinds=numpy.array(list(set(ridgeinds)|set(ridgeinds2)))
    if verbose:
        print 'ridge inds passed length tests: ', ridgeinds
        print ridges
    peaks=[]#list of [peak scaleind, posnind]
    mother_peaks=[] # every element is a tuple, the 1st elemnt is like an entry of peaks, the 2nd is a list of the children
    ridgeind_peaks=[]
    for count, ridge in enumerate(ridges):
        rind=numpy.where(ridge!=32767)[0]

        if len(rind)>0: #if this fails that means the ridge is essentially empty
            motherbool=ridge[rind[-1]]<0
            if verbose:
                if motherbool:
                    tempstr='(MOTHER) '
                else:
                    tempstr=''
                print 'NEW RIDGE ', tempstr, count, ': ', ridge
                print 'length test:', len(rind)>0
            rind=numpy.where((ridge!=32767)&(ridge>=0))[0] #this will generally be a continuous sequence of indeces except for possibler holes of size numscalesskippedinaridge
            wtvals=(wt[(wt.shape[0]-1-rind, ridge[rind])]) #-rind inverts but resulting order is still that of rind. wtvals is now the select values from wt but ordered from big->small wavelet scale
            totridgewt=wtvals.sum()
            ridgelen=len(rind)
            motherind=motherridgeind_childridge(ridge)
            if not motherind is None:
                if verbose:
                    print 'tot wt of child test:', totridgewt, minchildwtsum, totridgewt>minchildwtsum
                if totridgewt<=minchildwtsum:
                    continue
                if verbose:
                    print 'length of child test:', ridgelen, minchildlength, ridgelen>=minchildlength
                if ridgelen<minchildlength:
                    continue
                mridge=ridges[motherind]
                mrind=numpy.where((mridge!=32767)&(mridge>=0))[0] #this will generally be a continuous sequence of indeces except for possibler holes of size numscalesskippedinaridge
                mwtvals=(wt[(wt.shape[0]-1-mrind, mridge[mrind])])
                if verbose:
                    print 'mother ridge index: ', motherind, 'the ridge contributes ', mwtvals.sum(), " and is ", mridge
                totridgewt+=mwtvals.sum()
                ridgelen+=len(mrind) #if a child ridge has a mother that is the child of another mother, the wt and len from this grandmother does not count towards the total of the grandchild
            if verbose:
                print 'ridgelen test', ridgelen, minridgelength, ridgelen>=minridgelength
            if ridgelen>=minridgelength:
                indforlocalmaxtest=(rind>=ridgescalecritind) #if bigger ridge index, smaller wavelet scale (used to be called indforincreasingtest)
                #if (wtvals[indforincreasingtest][1:]>wtvals[indforincreasingtest][:-1]).sum()>0: #if a ridge wt value is bigger than its predecessor(larger scale) then the wt isn't strictly increasing with increasing qscale
                if verbose:
                    print 'the ridge indeces with wave scale less than critical:', indforlocalmaxtest
                    print 'any there test: ', len(wtvals[indforlocalmaxtest])>0
                    if len(wtvals[indforlocalmaxtest])>0:
                        print 'local max test: ', numpy.max(wtvals[indforlocalmaxtest])>=numpy.max(wtvals)
                if len(wtvals[indforlocalmaxtest])>0 and numpy.max(wtvals[indforlocalmaxtest])>=numpy.max(wtvals): #local max is at a scale smaller than critical index - this does nto include mother ridge - the large-scale part of the ridge got to count towards the ridge length
                    if verbose:
                        print 'tot wt test:',  totridgewt, minridgewtsum,  totridgewt>minridgewtsum
                    if totridgewt>minridgewtsum:
                        scaleind=(wt.shape[0]-1-rind[numpy.nanargmax(wtvals)]) #choose the wt scale and posn at the local maximum of wt - this does not include mother ridges. scaleind is now appropriate for wt (not ridges)
                        posnind=ridge[rind[-1]]#choose the position from the smallest scale in the ridge
                        if motherbool:
                            mother_peaks+=[(count, [scaleind, posnind], family_ridge(ridges, count)[2])]
                        else:
                            peaks+=[[scaleind, posnind]]
                            ridgeind_peaks+=[count]

    ind_potentialpeaks=set(ridgeind_peaks)
    for ind, pk, descendants in mother_peaks:
        ind_potentialpeaks|=set([ind])
    for ind, pk, descendants in mother_peaks:
        if len(ind_potentialpeaks&set(descendants))==0:
            peaks+=[pk]
            if verbose:
                print 'MOTHER RIDGE ',  ind, ' BECOMES PEAK:', pk, '. The non-peak children are indeces ', descendants
        else:
            if verbose:
                print 'MOTHER RIDGE ',  ind, ' NOT A PEAK BECAUSE OF EXISTENCE OF DESCENDANTS: ',  descendants

    return peaks
def pksort(arr):
    if arr.size==0:
        return numpy.uint32([[], []])
    sortind=arr[1].argsort()
    return numpy.uint32([arr[0, sortind], arr[1, sortind]])
def combinepeakarrays(peakset1, peakset2):
    return [pksort(numpy.append(pk1, pk2, axis=1)) for pk1, pk2 in zip(peakset1, peakset2)]
    
def peaks_ridges1d(wavetrset, ridgesset, scalevals, padfirstdimintoarray=True, minridgelength=3, minchildlength=0., maxscale_localmax=None, minridgewtsum=100., minchildwtsum=0., verbose=False): 
    """
    peaks_ridges1d(wavetrset, ridgesset, scalevals, padfirstdimintoarray=True, minridgelength=3, minchildlength=0., maxscale_localmax=None, minridgewtsum=100., minchildwtsum=0., verbose=False)
    wavetrset, ridgesset, scalevals are from previous function runs, see example.py
    minridgelength: minimum length (i.e. number of wavelet scales) of ridge for it to count as a peak
    minchildlength: same thing but for length of the child ridge alone, not count its mother's length
    maxscale_localmax: the largest scale in which the ridge can be locally maximum and still count as a peak
    minridgewtsum: critical value for the sum of the wavelet transform over the ridge. This will depend on the number of wavelet scales used and the nominal intensities of the original data.
    minchildwtsum: same thing except for the child ridge. The child ridge must have this total intensity to count as a peak, but for the "minridgewtsum", the child and mother intensities are added together.
    """
    minridgelength=max(1, minridgelength)

    ridgescalevals=scalevals[::-1] #ordered big->small
    if maxscale_localmax is None:
        maxscale_localmax=max(scalevals)
    ridgescalecritind=numpy.where(ridgescalevals<=maxscale_localmax)[0]
    if len(ridgescalecritind)<2:
        print 'aborted: the set of qscales does not include more than 1 point in the specified qwidthrange'
        return 'aborted: the set of qscales does not include more than 1 point in the specified qwidthrange'
    ridgescalecritind=ridgescalecritind[0]

    peaks=[perform_peaks_ridges1d(wt, ridges, ridgescalecritind=ridgescalecritind, minridgelength=minridgelength, minchildlength=minchildlength, minridgewtsum=minridgewtsum, minchildwtsum=minchildwtsum, verbose=verbose) for wt, ridges in zip(wavetrset, ridgesset)]
    
    numpks=[len(p) for p in peaks]
    maxnp=max(numpks)
    filler=[[32767]*2]*maxnp
    
    if padfirstdimintoarray:
        for p in peaks:
            p+=filler[:len(filler)-len(p)]
        return numpy.uint32([pksort(numpy.uint32(p).T) for p in peaks])
    else:
        return [pksort(numpy.uint32(p).T) for p in peaks]#list of 2 x numpeaks arrays,m the 2 are scaleind and posnind

def motherridgeind_childridge(ridge):#returns None if the ridge has no mother - assumes the ridge is indexed in decreasing order of qscale
    validridgeind=numpy.where((ridge!=32767))[0]
    negridgeind=numpy.where((ridge!=32767)&(ridge<0))[0]
    if len(negridgeind)>0 and negridgeind[0]==validridgeind[0]:#the second condition fails if this ridge is a mother ridge that is not the child of a different ridge
        return -1*ridge[negridgeind[0]]-1
    else:
        return None

def children_ridge(ridges, ind):#returns list of children - assumes the ridge is indexed in decreasing order of qscale
    ridge=ridges[ind]
    mothind=motherridgeind_childridge(ridge)
    if mothind is not None:
        mothset=set([mothind])
    else:
        mothset=set([])
    children=set(numpy.where(ridges==(-1*ind-1))[0])-mothset
    return sorted(list(children))

def family_ridge(ridges, ind):#returns mother (None if the ridge has no mother) and list of children - assumes the ridge is indexed in decreasing order of qscale
    ridge=ridges[ind]
    mothind=motherridgeind_childridge(ridge)

    children=children_ridge(ridges, ind)
    descendants=children
    generation=children
    while len(generation)>0:
        nextgeneration=[]
        for chind in generation:
            nextgeneration+=children_ridge(ridges, chind)
        generation=nextgeneration
        descendants+=nextgeneration

    return mothind,  children, sorted(descendants)


def expandwaveletscaleby2(waveset):#the overall shape compared to the scale gets slightly modified
    expandeddata=numpy.float32([numpy.float32([w, numpy.append((w[1:]+w[:-1])/2., [w[-1]])]).T.flatten()[:-1] for w in waveset])
    expandedposns=numpy.float32([numpy.float32([w, numpy.append((w[1:]+w[:-1])/2., [w[-1]])]).T.flatten()[:-1] for w in expandeddata.T])
    #the above line has an undesired shift in x. the error in position is nothing but the energy and zerosum have been violated. below, the zerosum is return and the energy is conserved within .5% up to 4th harmonic
    shifted=numpy.float32([w-(w.sum()/len(w)) for w in expandedposns.T])
    return shifted/numpy.sqrt(2.)
    


def getwavesetfromdatabase(scales, maxfixenfrac=(1./1.12, 1.12), path=WAVELETDATABASEPATH):
    """
    getwavesetfromdatabase(scales, maxfixenfrac=(1./1.12, 1.12), path=WAVELETDATABASEPATH)
    reads wavelets specified by scales from database and performs energy correction be assymetrically stretching the wavelet in the y-direction.
    If the wavelet energy is beyond the range specified by maxfixenfrac, this wavelet is marked as 'bad' and will not be used in the wavelet transform'
    """
    f=numpy.load(path)
    allscales=f['scales']
    nscales_side=f['nscales_side'][0]
    fixen=f['fixen']
    waves=f['waves']
    count=0
    #print allscales
    allscalesexpanded=[s for s in allscales]
    while max(allscalesexpanded)<max(scales):
        count+=1
        mult=2.**count
        allscalesexpanded+=[mult*s for s in allscales]
    allscalesexpanded=numpy.float32(allscalesexpanded)
    
    expandedscaleinds=[numpy.argmin((allscalesexpanded-s)**2) for s in scales]
    harmonics, scaleinds=zip(*[(i//len(allscales), i%len(allscales)) for i in expandedscaleinds])
    scaleinds=list(scaleinds)
    harmonics=list(harmonics)
    savesize_scales=numpy.uint64([nscales_side*s+1 for s in allscales])
    si=numpy.array([savesize_scales[:i].sum() for i in range(len(savesize_scales)+1)])
    fixen=[fixen[i:j] for i, j in zip(si[:-1][scaleinds], si[1:][scaleinds])]
    ws=(savesize_scales*2-1)*savesize_scales
    si=numpy.array([ws[:i].sum() for i in range(len(ws)+1)])
    waves=[waves[i:j].reshape(k, k*2-1) for i, j, k in zip(si[:-1][scaleinds], si[1:][scaleinds], savesize_scales[scaleinds])]
    #print harmonics, len(fixen)
    for count, (fe, h) in enumerate(zip(fixen, harmonics)):
        nani=numpy.where((fe<maxfixenfrac[0])|(fe>maxfixenfrac[1]))
        waves[count][nani]=numpy.nan
        for i in range(h):
            waves[count]=expandwaveletscaleby2(waves[count])
    waves=[numpy.concatenate([w[:-1], w[-1:], w[-2::-1,::-1]], axis=0) for w in waves]

    return waves, [allscalesexpanded[si] for si in expandedscaleinds]




class wavelet1dplotclass():
    """plotting class in which display_wavetrans1d is the main function.
    __init__ takes a parameter 'fig' which is a matplotlib.figure.Figure instance in which the plots will be made and a parameter 'scalegrid' which contains the scale values
    The top suplot will contains the data in blue and identified peaks in red. 
    The bottom plot will contains the wavelet trasform in a sqrt color scale. 
    Peaks in the wavelet transform at each scale value will be plotted as greyscale circles 
    where the greyscale values are the nromalized intensity within the ridge. 
    Ridges are connected by solid lines and mother-child ridges are connected by dotted lines.
    
    """
    def __init__(self, fig, qscalegrid, numposns=None, qgrid=None, qposngrid=None, showcolbar=True):#in simple case where wavelet posns are same as data posns just pass numposns, otherwise need to give qgrid and qposngrid
        #super(plotwidget, self).__init__(parent) #***
        #plotdata can be 2d array for image plot or list of 2 1d arrays for x-y plot or 2d array for image plot or list of lists of 2 1D arrays
        self.colbar=None
        self.showcolbar=showcolbar
        self.fig=fig
        axes=self.fig.add_subplot(211)
        axes.hold(False)
        axes=self.fig.add_subplot(212)
        axes.hold(False)

        if qgrid is None:
            self.qgrid=[0., 1., numposns]
        else:
            self.qgrid=qgrid
        if qposngrid is None:
            self.qposngrid=[0., 1., numposns]
        else:
            self.qposngrid=qposngrid
        
        self.qscalegrid=qscalegrid

        self.wtaxes=self.fig.add_subplot(212)

        self.xlim=(-.5, self.qgrid[2]-.5)
        
        self.scalelim=(-.5, self.qscalegrid[2]-.5)

        self.dataaxes=self.fig.add_subplot(211, sharex=self.wtaxes)

        self.fig.subplots_adjust(right=0.85)

        #self.mpl_connect('button_press_event', self.myclick)
    def myclick(self, event):
        if not ((event.xdata is None) or (event.ydata is None)):
            if event.inaxes==self.dataaxes:
                self.emit(SIGNAL("dataaxesclicked"), [event.xdata, event.ydata])
            if event.inaxes==self.wtaxes:
                self.emit(SIGNAL("wtaxesclicked"), [event.xdata, event.ydata])

    def display_wavetrans1d(self, wt, ridges, data, datascaleind, datapeakind=None, wtcmap=cm.jet, ridgecmap=cm.gray, ridgestyle=(('o', 15,0),('k-', 1.)), motherchildline_clrwidth=('k-.', 1.5),  title='', logdata=False):
        self.dataaxes.set_xticks([])
        self.dataaxes.set_yticks([])
        ridgewtscatter=[]
        for r in ridges:
            scaleinds=numpy.where((r!=32767)&(r>=0))
            posninds=r[scaleinds]
            scaleinds_increasingscale=wt.shape[0]-scaleinds[0]-1
            motherind=motherridgeind_childridge(r)
            if motherind is None:
                mothertochildplotcoords=None
            else:
                print 'MOTHERIND', motherind
                print r
                print len(ridgewtscatter), len(ridgewtscatter[motherind])
                print len(ridgewtscatter[motherind][0]), len(ridgewtscatter[motherind][1])
                xmother=ridgewtscatter[motherind][0][-1]
                ymother=ridgewtscatter[motherind][1][-1]
                xchild=posninds[0]
                ychild=scaleinds_increasingscale[0]
                mothertochildplotcoords=([xmother, xchild], [ymother, ychild])#assume that mother ridges always appear in "ridges" before the children ridges
            ridgewtscatter+=[(posninds, scaleinds_increasingscale, wt[(scaleinds_increasingscale, posninds)], mothertochildplotcoords)]
        if datascaleind is None:
            qpind_data=ind_qgrid_q(self.qposngrid, q_qgrid_ind(self.qgrid), fractional=True)
            colstr='b'
        else:
            qpind_data=numpy.array(range(int(round(self.qposngrid[2]))))
            colstr='g'

        aspect=.3*self.qposngrid[2]/self.qscalegrid[2]
        #vmin, vmax=numpy.min(wt[numpy.logicalnotnumpy.isnan(wt)

        maskedwt = ma.masked_where(numpy.isnan(wt), numpy.sign(wt)*numpy.sqrt(numpy.abs(wt)))


        self.wtaxes.clear()

        if self.colbar is None or self.im is None:
            self.im=self.wtaxes.imshow(maskedwt, origin='lower', aspect=aspect, cmap=wtcmap, interpolation='nearest')
        else:
            actuallyshownim=self.wtaxes.imshow(maskedwt, origin='lower', aspect=aspect, cmap=wtcmap, interpolation='nearest')
            self.im.set_data(maskedwt)
            self.im.set_clim(maskedwt.min(), maskedwt.max())
            self.im.set_cmap(wtcmap)
            self.colbar.set_cmap(wtcmap)
            self.im.changed()

        if self.colbar is None:
            self.colbaraxes=self.fig.add_axes([0.9, 0.1, 0.02, 0.3])
            self.colbar=self.fig.colorbar(self.im, cmap=wtcmap, cax=self.colbaraxes)

        self.wtaxes.set_xlabel('wavelet position')
        self.wtaxes.set_ylabel('wavelet scale')
        self.wtaxes.hold(True)

        for x, y, c, motherchild in ridgewtscatter:
            if ridgecmap is None:
                self.wtaxes.plot(x, y, ridgestyle[0][0]+ridgestyle[1][0], markersize=ridgestyle[0][1], linewidth=ridgestyle[1][1])
            else:
                if ridgestyle[1][0]!='':
                    self.wtaxes.plot(x, y, ridgestyle[1][0], linewidth=ridgestyle[1][1])
                self.wtaxes.scatter(x, y, c=c, cmap=ridgecmap, s=ridgestyle[0][1], linewidth=ridgestyle[0][2])
                
            if not (motherchild is None) and not (motherchildline_clrwidth is None):
                self.wtaxes.plot(motherchild[0], motherchild[1], motherchildline_clrwidth[0], linewidth=motherchildline_clrwidth[1])

        self.dataaxes.clear()
        self.dataaxes.hold(False)

        datainds=numpy.where(numpy.logical_not(numpy.isnan(data)))
        self.dataaxes.plot(qpind_data[datainds], data[datainds], colstr, linewidth=3)
        self.dataaxes.hold(True)

        if not (datapeakind is None):
            if datascaleind is None:
                for x, y in zip(ind_qgrid_q(self.qposngrid, q_qgrid_ind(self.qgrid, datapeakind), fractional=True), data[(numpy.uint32(numpy.round(datapeakind)), )]):
                    self.dataaxes.plot([x, x], [data[numpy.logical_not(numpy.isnan(data))].min(), y], 'r')
            else:
                self.dataaxes.plot(datapeakind, data[(numpy.uint32(numpy.round(datapeakind)), )], 'k*',  markersize=11)

        if logdata:
            self.dataaxes.set_yscale('log')
        else:
            self.dataaxes.set_yscale('linear')

        self.dataaxes.hold(False)
        if datascaleind is None:
            self.dataaxes.set_xlabel('data position')
            self.dataaxes.set_ylabel('data value')
        else:
            self.dataaxes.set_xlabel('wavelet position')
            self.dataaxes.set_ylabel('wavelet transform at scale %.2f' %scale_scalegrid_ind(self.qscalegrid, datascaleind))

        qslabelind=sorted(list(set(numpy.int32(numpy.linspace(0., (self.qscalegrid[2]-1.), 5)))))
        qslabels=['%.1f' %qs for qs in scale_scalegrid_ind(self.qscalegrid, qslabelind)]

        self.wtaxes.set_yticks(qslabelind)
        self.wtaxes.set_yticklabels(qslabels)

        self.dataaxes.set_xlim(self.xlim)
        self.wtaxes.set_xlim(self.xlim)
        self.wtaxes.set_ylim(self.scalelim)

    def plot1doverlay(self, data, datascaleind, datapeakind=None):
        self.dataaxes.hold(True)
        if datascaleind is None:
            qpind_data=ind_qgrid_q(self.qposngrid, q_qgrid_ind(self.qgrid), fractional=True)
            colstr='b'
        else:
            qpind_data=numpy.array(range(int(round(self.qposngrid[2]))))
            colstr='g'

        self.dataaxes.plot(qpind_data, data, colstr)

        if not (datapeakind is None):
            if datascaleind is None:
                for x, y in zip(ind_qgrid_q(self.qposngrid, q_qgrid_ind(self.qgrid, datapeakind), fractional=True), data[(numpy.uint32(numpy.round(datapeakind)), )]):
                    self.dataaxes.plot([x, x], [data.min(), y], 'r')
            else:
                self.dataaxes.plot(datapeakind, data[(numpy.uint32(numpy.round(datapeakind)), )], 'k+')

        if datascaleind is None:
            self.dataaxes.set_xlabel('data position')
            if 'wavelet' in self.dataaxes.get_ylabel():
                self.dataaxes.set_ylabel('wavelet and data intensity')
        else:
            if 'wavelet' in self.dataaxes.get_ylabel():
                self.dataaxes.set_ylabel(self.dataaxes.get_ylabel() +', %.2f /nm' %scale_scalegrid_ind(self.qscalegrid, datascaleind))
            else:
                self.dataaxes.set_ylabel('wavelet and data intensity')

    def save(self, name):
        self.fig.savefig(''.join((name, '.png')), dpi=300)

def filterpeaksbyfirstder(data, pkset, critderval, SGnptsoneside=20, SGorder=2, removeslopebaselinewindow=10):#if slope is too, high, don't count as peak
    deratpeaks=[savgolsmooth(d, nptsoneside=SGnptsoneside, order=SGorder, deriv=1)[pk[1]] for d, pk in zip(data, pkset)]
    if removeslopebaselinewindow>0:
        aveslope=[(d[-1*removeslopebaselinewindow:].mean()-d[:removeslopebaselinewindow].mean())/(len(d)-removeslopebaselinewindow) for d in data]
    else:
        aveslope=[0.]*len(data)
    goodpkinds=[numpy.where(numpy.abs(der-a)<critderval) for der, a in zip(deratpeaks, aveslope)]
    return [numpy.uint32([pk[0][i], pk[1][i]]) for pk, i in zip(pkset, goodpkinds)]
    
def filterpeaksbysecder(data, pkset, critderval, SGnptsoneside=20, SGorder=2):#if slope is too, high, don't count as peak
    deratpeaks=[savgolsmooth(d, nptsoneside=SGnptsoneside, order=SGorder, deriv=2)[pk[1]] for d, pk in zip(data, pkset)]
    print 'secer', deratpeaks
    goodpkinds=[numpy.where(der<critderval) for der in deratpeaks]
    return [numpy.uint32([pk[0][i], pk[1][i]]) for pk, i in zip(pkset, goodpkinds)]
    
