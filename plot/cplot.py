import matplotlib.pyplot as plt
import scipy
import scipy.io
from matplotlib import rc
import sys
import dd
sys.path.append('/afs/ipp-garching.mpg.de/home/i/ianf/codes/python/TRIPPy-master')
sys.path.append('/afs/ipp-garching.mpg.de/home/i/ianf/codes/python/general')
import plotEq
import sqlite3

#import TRIPPy.beam
#import TRIPPy.geometry
#import GIWprofiles
#import SIFweightings

import _gauss as _gauss2

from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

rc('text',usetex=True)
rc('text.latex',preamble='\usepackage{xcolor}')
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
#rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern Sans serif']})
rc('font',size=20)

import eqtools

data = scipy.io.readsav('W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave')
#2017 calibration off of 34228 w line
calib = 33636.23 #33354.6# 33636.23
disp = 86363.6363 #85650.6# 86363.6363
semiEmperical = scipy.array([.3909,.39316,.39331,.39374,.39492,.3955,.3963,.39648,.3966,.39695,.3975,.39887,.39943])
semiNames=['W$^{44+}$',' ','W$^{45+}$','?','(W)',' ',' ','W$^{46+?}$','(X)','(Y)','W$^{44+}$','W$^{44+}$','(Z)']

def genSIF(tokamak):
    """ Use the TRIPPy system to yield the spectrometer sightline """

    #set up two points for extracting
    vec1 = TRIPPy.geometry.Vecr([1.785,0.,1.08])
    pt1 = TRIPPy.geometry.Point(vec1,tokamak)

    vec2 = TRIPPy.geometry.Vecr([1.785,0.,1.06])
    pt2 = TRIPPy.geometry.Point(vec2,tokamak)
    
    beam = TRIPPy.beam.Ray(pt1,pt2)
    tokamak.trace(beam) #modify beam to yield the proper in and exit points in the tokamak

    return beam #this contains the functionality to do the analysis.

def genJMO(tokamak):
    """ Use the TRIPPy system to yield the spectrometer sightline """

    #set up two points for extracting
    vec1 = TRIPPy.geometry.Vecr([2.194,292.500/180.*scipy.pi,-.008])
    pt1 = TRIPPy.geometry.Point(vec1,tokamak)

    vec2 = TRIPPy.geometry.Vecr([1.060,309.829*scipy.pi/180.,.246])
    pt2 = TRIPPy.geometry.Point(vec2,tokamak)
    
    beam = TRIPPy.beam.Ray(pt1,pt2)
    #tokamak.trace(beam) #modify beam to yield the proper in and exit points in the tokamak

    return beam #this contains the functionality to do the analysis.


def genInp(num=1e-3,shot=34228):
    tok = TRIPPy.Tokamak(eqtools.AUGDDData(34228))
    ray = genSIF(tok)
    print(ray.norm.s)
    #inp = scSipy.linspace(ray.norm.s[-2],ray.norm.s[-1],num)
    inp = scipy.mgrid[ray.norm.s[-2]:ray.norm.s[-1]:num]
    r = ray(inp).r()[0]
    z = ray(inp).r()[2]
    l = inp - inp[0]
    return [r,z,l]


def SIFData(shot, filename="SIF", data="DataSIF", offset=0, rev=True):

    shotfile = dd.shotfile(filename,shot)
    temp = shotfile(data)
    temp.data = temp.data[:,:1024]
    temp.data = temp.data[:,::-1] #flip axis for wavelength data
    #temp.wavelength = (scipy.arange(temp.data.shape[1]) + offset + calib)/dispS
    #temp.wavelength = temp.wavelength[:1024]
    temp.wavelength = (scipy.arange(1250) - 34334.5 + offset)/(-85650.6) #give it a wavelength parameter.  This is hardcoded and derived from Sertoli's work in /u/csxr/sif/sif.pro
    temp.wavelength = temp.wavelength[:1024]
    temp.wavelength = temp.wavelength[::-1]
    return temp #send it out into the wild

def SIFData2(shot, filename="SIF", data="DataSIF", offset=0, rev=True):

    shotfile = dd.shotfile(filename,shot)
    temp = shotfile(data)
    temp.data = temp.data[:,:1024]
    temp.data = temp.data[:,::-1] #flip axis for wavelength data
    temp.wavelength = (scipy.arange(temp.data.shape[1]) + offset + calib)/disp
    temp.wavelength = temp.wavelength[:1024]
    return temp #send it out into the wild

def genGRW(tokamak):
    """ Use the TRIPPy system to yield the spectrometer sightline """

    #set up two points for extracting
    vec1 = TRIPPy.geometry.Vecr([2.200,scipy.pi*11.25/180.,.129])
    pt1 = TRIPPy.geometry.Point(vec1,tokamak)

    vec2 = TRIPPy.geometry.Vecr([1.050,scipy.pi*11.25/180.,-.029])
    pt2 = TRIPPy.geometry.Point(vec2,tokamak)

    #I really don't like these points, and I might do some trickery to yield better data
    
    beam = TRIPPy.beam.Ray(pt1, pt2)

    pt3 = beam(scipy.array([-1.]))
    pt4 = beam(scipy.array([-.8]))

    temp = TRIPPy.geometry.Point(pt3,tokamak)

    beam2 = TRIPPy.beam.Ray(pt3.point(tokamak), pt4.point(tokamak))

    tokamak.trace(beam2) #modify beam to yield the proper in and exit points in the tokamak
    return beam2


def genInp2(num=1e-3,shot=34228):
    tok = TRIPPy.Tokamak(eqtools.AUGDDData(34228))
    ray = genGRW(tok)
    print(ray.norm.s)
    #inp = scipy.linspace(ray.norm.s[-2],ray.norm.s[-1],num)
    inp = scipy.mgrid[ray.norm.s[-2]:ray.norm.s[-1]:num]
    r = ray(inp).r()[0]
    z = ray(inp).r()[2]
    l = inp - inp[0]
    return [r,z,l]


def plotData():
    print(data['abundance'].shape)
    for i in xrange(len(data['abundance'].T)):
        if (i >= 27) and (i <= 35) or (i >= 41) and (i <= 45):
            plt.loglog(data['en'], data['abundance'][:,i], color='#A02C2C', lw=2)
        else:
            plt.loglog(data['en'], data['abundance'][:,i], color='#0E528A', lw=2, alpha=.5)

    plt.loglog(data['en'], data['abundance'][:,46], color='k', lw=2)
    plt.annotate('W$^{27+}$-W$^{35+}$ (GI)',xy=(1.5e3,.175),xytext=(3e2,.4),arrowprops={'facecolor':'#A02C2C','shrink':.025})

    plt.annotate('W$^{41+}$-W$^{45+}$ (GI)',xy=(3.e3,.2),xytext=(7e2,.75),arrowprops={'facecolor':'#A02C2C','shrink':.025})

    plt.annotate('W$^{46+}$',xy=(4.8e3,.35),xytext=(6e3,.75),arrowprops={'facecolor':'k','shrink':.025})
    plt.xlim(1e2,1e4)
    plt.ylim(1e-2,1)
    plt.xlabel('$T_e$ [eV]')
    plt.ylabel('fractional abundance')
    plt.subplots_adjust(bottom=.12,top=.92)
    plt.show()

def plotCross(shot,time):
    eq = eqtools.AUGDDData(shot)
    eq.remapLCFS() #get that pretty boundary
    rl = eq.getRLCFS()[eq._getNearestIdx(eq.getTimeBase(),time)]
    zl = eq.getZLCFS()[eq._getNearestIdx(eq.getTimeBase(),time)]
    out = eq.getMachineCrossSectionFull()
    plt.plot(rl,zl,lw=2)
    plt.plot(out[0],out[1],'k')
    plt.gca().set_aspect('equal')
    plt.gca().autoscale(tight=True)
    plt.show()



def plotSIFAr(shot=24146,tidx=272,tidx2=10,offset=50):
    data = SIFData(shot,offset=offset)

    if tidx2 != 0:
        y = scipy.sum(data.data[tidx:tidx+tidx2],axis=0)
        delt= data.time[tidx+tidx2] - data.time[tidx]
        plt.plot(data.wavelength,y/delt/1e3,lw=2,color='#228b22')

    else:
        plt.plot(data.wavelength,data.data[tidx])

    print(data.time[tidx],data.time[tidx+tidx2])
    plt.xlim([.388,.4])
    
    temp = plt.axis()
    plt.fill_between([.3945,.4],[temp[-1],temp[-1]],alpha=.1,color='#0E525A')
    plt.annotate('area of interest',xy=(.3945,2000),xytext=(.389,3000),arrowprops={'facecolor':'k','shrink':.025,'width':1,'headwidth':6})
    plt.text(0.39492,10800,'W',fontsize=24)
    plt.text(0.3966,2200,'X',fontsize=24)
    plt.text(0.39695,2700,'Y',fontsize=24)
    plt.text(0.39943,3500,'Z',fontsize=24)
    plt.xlabel('wavelength [nm]')
    plt.ylabel('kCounts/s')
    plt.subplots_adjust(left=.14,bottom=.12,top=.92,right=.92)    
    limy = plt.gca().get_ylim()
    limx = plt.gca().get_xlim()
    plt.gca().text(1.0075*(limx[1]-limx[0])+limx[0],.97*(limy[1]-limy[0])+limy[0],str(shot),rotation=90,fontsize=14)
    #plt.show()

    
def plotSIFAr2(shot=33349,tidx=102,tidx2=10,offset=0):

    offsetx = scipy.array([0.,0.,0.,0.,0.,0.,-.00075,0.,0.001,0.0001,.0003,0.,0.])
    yval = scipy.array([2.7e3,4.75e3,5e3,1.9e3,5.2e3,0.,3.6e3,9.6e3,6e3,2.9e3,2.2e3,3.5e3,2.5e3])

    data = SIFData2(shot,offset=offset)


    plt.fill_between([.39459,.39525],[720.5,720.5],y2=[4950,4950],alpha=.2,color='#A02C2C')
    plt.fill_between([.39644,.39703],[720.5,720.5],y2=[2550,2550],alpha=.2,color='#A02C2C')
    plt.fill_between([.39913,.39961],[720.5,720.5],y2=[1890,1890],alpha=.2,color='#A02C2C')

    if tidx2 != 0:
        y = scipy.sum(data.data[tidx:tidx+tidx2],axis=0)
        delt= data.time[tidx+tidx2] - data.time[tidx]
        plt.plot(data.wavelength,y/delt/1e3,lw=2,color='#228b22')

    else:
        plt.plot(data.wavelength,data.data[tidx])

    print(data.time[tidx],data.time[tidx+tidx2])
    plt.xlim([.389,.401])
    plt.ylim([0,12000])
    temp = plt.axis()
    #plt.fill_between([.3945,.4],[temp[-1],temp[-1]],alpha=.1,color='#0E525A')
    #plt.annotate('area of interest',xy=(.3945,2000),xytext=(.389,3000),arrowprops={'facecolor':'k','shrink':.025,'width':1,'headwidth':6})

    for i in xrange(len(semiEmperical)):
        plt.text(semiEmperical[i]+offsetx[i],yval[i],semiNames[i],horizontalalignment='center',fontsize=24)
    plt.annotate(' ',xy=(semiEmperical[8],4000),xytext=(semiEmperical[8]+offsetx[8],yval[8]),arrowprops={'facecolor':'k','shrink':.05,'width':1,'headwidth':6})



    plt.xlabel('wavelength [nm]')
    plt.ylabel('kCounts/s')
    plt.subplots_adjust(left=.14,bottom=.12,top=.92,right=.92)    
    limy = plt.gca().get_ylim()
    limx = plt.gca().get_xlim()
    plt.gca().text(1.0075*(limx[1]-limx[0])+limx[0],.97*(limy[1]-limy[0])+limy[0],str(shot),rotation=90,fontsize=14)
    plt.show()


def plotSIFGI():

    a = eqtools.EqdskReader(gfile='/afs/ipp-garching.mpg.de/home/i/ianf/codes/python/general/g33669.3000')
    a.remapLCFS()
    b = eqtools.AUGDDData(33669)
    tok = TRIPPy.Tokamak(b)
    cont = (pow(scipy.linspace(0,1,11),2)*(a.getFluxLCFS()-a.getFluxAxis())+a.getFluxAxis())

    print(a.getFluxAxis())
    print(a.getFluxLCFS())
    print(cont)
    temp = b.getMachineCrossSectionFull()
    plotEq.plot(a,33669,lims=temp,contours=-1*cont)
    
    GI = genGRW(tok)
    r = GI.r()[0][-2:]
    z = GI.r()[2][-2:]
    print(r,z)
    plt.plot(r,z,color='#0E525A',linewidth=2.)

    SIF = genSIF(tok)
    r = SIF.r()[0][-2:]
    z = SIF.r()[2][-2:]
    print(r,z)
    plt.plot(r,z,color='#228B22',linewidth=2.)
    plt.subplots_adjust(left=.2,right=.95)
    plt.gca().set_xlim([scipy.nanmin(temp[0].astype(float)),scipy.nanmax(temp[0].astype(float))])
    plt.gca().set_ylim([scipy.nanmin(temp[1].astype(float)),scipy.nanmax(temp[1].astype(float))])
    plt.text(1.4,.9,'CSXR',fontsize=22,zorder=10)
    plt.text(2.25,.1,'GI',fontsize=22,zorder=10)
    limy = plt.gca().get_ylim()
    limx = plt.gca().get_xlim()
    plt.gca().text(1.0075*(limx[1]-limx[0])+limx[0],.97*(limy[1]-limy[0])+limy[0],str(33669),rotation=90,fontsize=14)


def plotSIFGI2():

    a = eqtools.EqdskReader(gfile='/afs/ipp-garching.mpg.de/home/i/ianf/codes/python/general/g33669.3000')
    a.remapLCFS()
    b = eqtools.AUGDDData(33669)
    tok = TRIPPy.Tokamak(b)
    cont = (pow(scipy.linspace(0,1,11),2)*(a.getFluxLCFS()-a.getFluxAxis())+a.getFluxAxis())

    print(a.getFluxAxis())
    print(a.getFluxLCFS())
    print(cont)
    temp = b.getMachineCrossSectionFull()
    plotEq.plot(a,33669,lims=temp,contours=-1*cont[::-1])
    
    GI = genGRW(tok)
    r = GI.r()[0][-2:]
    z = GI.r()[2][-2:]
    print(r,z)
    plt.plot(r,z,color='#0E525A',linewidth=2.)

    SIF = genSIF(tok)
    r = SIF.r()[0][-2:]
    z = SIF.r()[2][-2:]
    print(r,z)
    plt.plot(r,z,color='#228B22',linewidth=2.)


    JMO = genJMO(tok)
    r = JMO.r()[0][-2:]
    z = JMO.r()[2][-2:]
    print(r,z)
    plt.plot(r,z,color='#800080',linewidth=2.)


    plt.subplots_adjust(left=.2,right=.95)
    plt.gca().set_xlim([scipy.nanmin(temp[0].astype(float)),scipy.nanmax(temp[0].astype(float))])
    plt.gca().set_ylim([scipy.nanmin(temp[1].astype(float)),scipy.nanmax(temp[1].astype(float))])
    plt.text(1.4,.9,'CSXR',fontsize=22,zorder=10)
    plt.text(2.25,.1,'GI',fontsize=22,zorder=10)
    plt.text(2.2,-.1,'JMO',fontsize=22,zorder=10)
    limy = plt.gca().get_ylim()
    limx = plt.gca().get_xlim()
    plt.gca().text(1.0075*(limx[1]-limx[0])+limx[0],.97*(limy[1]-limy[0])+limy[0],str(33669),rotation=90,fontsize=14)

def plotSIF2(num=1e-4):

    a = eqtools.EqdskReader(gfile='/afs/ipp-garching.mpg.de/home/i/ianf/codes/python/general/g33669.3000')
    a.remapLCFS()
    b = eqtools.AUGDDData(33669)
    tok = TRIPPy.Tokamak(b)
    cont = (pow(scipy.linspace(0,1,11),2)*(a.getFluxLCFS()-a.getFluxAxis())+a.getFluxAxis())

    print(a.getFluxAxis())
    print(a.getFluxLCFS())
    print(cont)
    temp = b.getMachineCrossSectionFull()
    plotEq.plot(a,33669,lims=temp,contours=-1*cont[[0,10]],alpha=.15)

    ray = genSIF(tok)
    print(ray.norm.s)
    #inp = scipy.linspace(ray.norm.s[-2],ray.norm.s[-1],num)
    inp = scipy.mgrid[ray.norm.s[-2]:ray.norm.s[-1]:num]
    r = ray(inp).r()[0]
    z = ray(inp).r()[2]
    l = inp - inp[0]

    print(r.shape,z.shape,l.shape)

    plt.plot(r,z,color='#228B22',linewidth=2.)

    plt.subplots_adjust(left=.2,right=.95)
    plt.gca().set_xlim([scipy.nanmin(temp[0].astype(float)),scipy.nanmax(temp[0].astype(float))])
    plt.gca().set_ylim([scipy.nanmin(temp[1].astype(float)),scipy.nanmax(temp[1].astype(float))])
    limy = plt.gca().get_ylim()
    limx = plt.gca().get_xlim()
    plt.gca().text(1.0075*(limx[1]-limx[0])+limx[0],.97*(limy[1]-limy[0])+limy[0],str(33669),rotation=90,fontsize=14)
    
    ax1 = plt.gca()
    ax1.arrow(1.8,-.608,.025,0.0,head_width=.05,head_length=.1,fc='k',ec='k',zorder=10)
    ax1.arrow(1.8,0.728,.025,0.0,head_width=.05,head_length=.1,fc='k',ec='k',zorder=10)

    iax = plt.axes([0,0,1,1])

    x0 = (r.min()-limx[0])/(limx[1]-limx[0])
    y0 = (z.min()-limy[0])/(limy[1]-limy[0])
    y1 = (z.max()-limy[0])/(limy[1]-limy[0])

    print(limx)
    print(limy)

    print([x0+.05,y0,.2,y1-y0])

    ip = InsetPosition(ax1, [x0+.1,y0,.3,y1-y0])
    #ip = plt.axes([.05,.05,.02,.02])
    iax.set_axes_locator(ip)
    

    tempData = scipy.squeeze(GIWprofiles.te(b.rz2psinorm(r,z,[3.0],sqrt=True,each_t=True),4e3,1e3))
    print(tempData.shape)
    iax.semilogx(tempData,l,color='#0E525A',linewidth=2.)
    iax.set_ylim(l.max(),l.min())
    iax.set_xlim(1e2,1e4)
    iax.set_xlabel('$T_e$ [eV]')
    iax.text(2e2,1.0,'$l(T_e)$')
    iax.yaxis.tick_right()
    iax.set_axis_bgcolor('#eaeaea')


def pltne(nped=.7e20,ncor=1e20,tped=1e3,tcor=4e3,num=1e-3):
    fig = plt.figure(figsize=(6*.8,11*.6))
    ax = fig.add_subplot(211)
    te = data['en']

    ne = GIWprofiles.ne(GIWprofiles.te2rho2(te,4e3,1e3),1.0e20,.7e20)

    ax.semilogx(te,ne/1e19,color='#0E525A',linewidth=2.)
    ax.set_xlim(1e2,1e4)
    ax.set_ylim(0.,12)
    ax.set_ylabel('$n_e$ [$10^{19}$ m$^{-3}$]') 
    ax.text(2e2,7.5,'$n_e(T_e)$')

    ax2 = fig.add_subplot(212,sharex=ax)

    ax2.set_ylabel('$n_e^2dl$ [$10^{38}$ m$^{-5}$]')

    plt.subplots_adjust(hspace=.2,left=.2,right=.95,wspace=.1)

    a = eqtools.EqdskReader(gfile='/afs/ipp-garching.mpg.de/home/i/ianf/codes/python/general/g33669.3000')

    tok = TRIPPy.Tokamak(a)
    ray = genSIF(tok)
    print(ray.norm.s)
    inp = scipy.mgrid[ray.norm.s[-2]:ray.norm.s[-1]:num]
    r = ray(inp).r()[0]
    z = ray(inp).r()[2]
    l = inp - inp[0]

    logte = scipy.log(te)
    halfte = scipy.exp(logte[1:]/2. + logte[:-1]/2.)
    rho = scipy.squeeze(a.rz2psinorm(r, z, each_t=True, sqrt=True)) 
    print(rho.shape,l.shape)
    #solve at each time 
    output = SIFweightings.calcArealDens(l, te, halfte, rho, tped, tcor, nped, ncor)
    ax2.semilogx(te,output/1e38,color='#0E525A',linewidth=2.)
    ax2.set_xlabel('$T_e$ [eV]')
    ax2.text(2e2,2,'$M(T_e)$')
    
    ax.set_axis_bgcolor('#eaeaea')



def plotShotData(shot):
    temp = sqlite3.connect('SIFdata5.db')
    b = temp.cursor()
    
    c_W = scipy.squeeze((b.execute('SELECT c_W from shots2016 where shot='+str(shot))).fetchall())
    c_W_l = scipy.squeeze((b.execute('SELECT c_W_l from shots2016 where shot='+str(shot))).fetchall())
    val7 = scipy.squeeze((b.execute('SELECT val7 from shots2016 where shot='+str(shot))).fetchall())
    time = scipy.squeeze((b.execute('SELECT time from shots2016 where shot='+str(shot))).fetchall())

    plt.legend(loc=2,fontsize=20)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.semilogy(time,val7/1e3,color='#0E525A',lw=2,label='CSXR line')


    ax2.semilogy(time,c_W_l,color='#A02C2C',lw=2,label='$c_W$ high $T_e$')

    ax2.semilogy(time,c_W,color='#228B22',lw=2,label='$c_W$ low $T_e$')
    #ax1.set_ylim(1e-6,1e-2)

    
    plt.legend(loc=2)
    ax1.set_ylim(1e-2,1e2)

    ax1.set_xlabel('time [s]')
    ax2.set_ylabel('$c_W$')
    ax1.set_ylabel('line intensity [kCounts]')
    plt.subplots_adjust(bottom=.12,right=.88) 
    limy = plt.gca().get_ylim()
    limx = plt.gca().get_xlim()
    plt.gca().text(.9*(limx[1]-limx[0])+limx[0],1.01*(limy[1]-limy[0])+limy[0],str(shot),rotation=0,fontsize=14)
    plt.show()


########################################################
#                 fit data comparison                  #
########################################################

def interface4(inp, xdata, ydata):

    p2 = _gauss2.gpmle(xdata,ydata,inp)
    #p = scipy.zeros((len(inp)+1,))
    #p2 = compareln(inp, xdata, ydata)
    #p[1:] = _approx_fprime_helper(inp,compare,1e-8,args=(xdata,ydata))
    #print('new')
    #print(inp)
    #print(p)
    #print(p2)
    if scipy.any(scipy.isnan(p2)):
        print(inp,p2,'error4')
    return p2,p2[1:]


def _fitData8(idx, xdata, data, bounds=None, method=None, tol=None):

    ydata =data[idx] - 1e3
    # need to subtract baseline from data


    if not bounds == None:
        inp, bounds = _assembleInit2(xdata, ydata, bounds=bounds)
    else:
        inp = _assembleInit2(xdata, ydata)
    #inp[0] *= .9 
    inp[0] /= 123.
    inp[1::3] /=123.
        
    output = scipy.optimize.minimize(interface4,
                                     inp,
                                     args=(xdata,ydata/123.),
                                     method=method,
                                     bounds=bounds,
                                     tol=tol,
                                     jac=True).x

    output2 = output[0] + _gauss2.gauss2(xdata, output[1:])
    print(output[1::3]*123.,'c code')
   
    return output2*123. + 1e3 #The 123 is a conversion of the Andor guide for counts to photons for .4nm light (E/25.55 in keV)

def _assembleInit2(xdata, ydata, bounds=None):

    #internal fitting bounds and initial value for baseline and gaussian height, offset, and width
    _bounds = scipy.array([[1,65536],[0,1e15],[-3e-5,3e-5],[5e-5,3e-4]])
    _bounds2 = scipy.array([[0,scipy.inf],[0,scipy.inf],[-3e-5,3e-5],[0,scipy.inf]])
    
    _initial = scipy.array([0.,1.,1.,.7e-4,0.])

    #INTITIALIZE VARIABLES
    init = scipy.zeros((3*len(semiEmperical)+1,))
    output = scipy.zeros((len(init), 2))
    ones = scipy.ones(semiEmperical.shape)
    
    #set baseline
    axis = scipy.mgrid[-32:65569:64]
    init[0] = (scipy.histogram(ydata,bins=axis)[0]).argmax()*64#ydata.min()#_initial[0]
    output[0] = _bounds[0]

    #set peak values
    print(ydata[scipy.searchsorted(xdata, semiEmperical)])
    init[1::3] = ydata[scipy.searchsorted(xdata, semiEmperical)] - init[0]
    init[1::3] = (abs(init[1::3]) + init[1::3])/2.
    output[1::3] = scipy.array([_bounds[1,0]*ones, _bounds[1,1]*ones]).T

    #set offsets
    init[2::3] = semiEmperical
    output[2::3] = (scipy.array([_bounds[2,0]*ones, _bounds[2,1]*ones]) + semiEmperical).T

    #set width values
    init[3::3] = _initial[3]*ones
    output[3::3] = scipy.array([_bounds[3,0]*ones, _bounds[3,1]*ones]).T

    init[1::3] *= init[3::3]*scipy.sqrt(scipy.pi) #convert to integrated counts
    output[1::3] *= output[3::3]*scipy.sqrt(scipy.pi) #same for these values
    print(init[1::3],'initial vals')

    if not bounds == None:
        return init, output
    else:
        return init
    
def plotSIFAr3(shot=33349,tidx=105,tidx2=10,offset=0):

    offsetx = scipy.array([0.,0.,0.,0.,0.,0.,-.00075,0.,0.001,0.0001,.0003,0.,0.])
    yval = scipy.array([2.7e3,4.75e3,5e3,1.9e3,5.2e3,0.,3.6e3,9.6e3,6e3,2.9e3,2.2e3,3.5e3,2.5e3])

    data = SIFData2(shot,offset=offset)

    if tidx2 != 0:
        y = scipy.sum(data.data[tidx:tidx+tidx2],axis=0)
        delt= data.time[tidx+tidx2] - data.time[tidx]
        plt.plot(data.wavelength,y/delt/1e3,lw=2,color='#228b22')

    else:
        plt.plot(data.wavelength,data.data[tidx]/1e3,color='#228b22',lw=2,alpha=.75,label='raw')
        output2 = _fitData8(tidx, data.wavelength, data.data, bounds=True)
        plt.plot(data.wavelength,output2/1e3,lw=2,color='#0E525A',label='fit')

    plt.xlim([.389,.401])
    plt.ylim([0,70])
    temp = plt.axis()

    plt.xlabel('wavelength [nm]')
    plt.ylabel('kCounts/s')
    plt.subplots_adjust(left=.14,bottom=.12,top=.92,right=.92)    
    limy = plt.gca().get_ylim()
    limx = plt.gca().get_xlim()
    plt.gca().text(1.0075*(limx[1]-limx[0])+limx[0],.97*(limy[1]-limy[0])+limy[0],str(shot),rotation=90,fontsize=14)
    plt.legend()
    plt.show()
