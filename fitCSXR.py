import matplotlib.pyplot as plt
import scipy.optimize
import scipy.io
#import sklearn
#import sklearn.linear_model
import sqlite3
import time
import testM
from matplotlib.colors import LogNorm
from matplotlib import rc
import adsre


rc('text',usetex=True)
rc('text.latex',preamble='\usepackage{xcolor}')
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
#rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern Sans serif']})
rc('font',size=20)

def gaussian(x,inp):
    return inp[0]*scipy.exp(-1*pow((x-inp[1])/inp[2],2))

def lognorm(x,inp):
    return (inp[0]/x)*scipy.exp(-1*pow((scipy.log(x)-inp[1])/inp[2],2))

def genI(inp, Te, c_W, M):
    return c_W*scipy.dot(M, lognorm(Te, inp))

def objectiveLog(inp, val, Te, c_W, M):
    return scipy.sqrt(scipy.sum(pow(scipy.log(val) - scipy.log(genI(inp, Te, c_W, M)),2)))

def objective(inp, val, Te, c_W, M):
    return scipy.sqrt(scipy.sum(pow(val - genI(inp, Te, c_W, M),2)))


def loadData(name, database='SIFdata6.db'):
    a = sqlite3.connect(database)
    b = a.cursor()
    return scipy.squeeze(b.execute('SELECT '+name+' FROM shots2016').fetchall())

def conditionVal(name='val7', nozero=False, conc='c_w_l'):
    #this is a program which extracts the useful data from the CSXR data

    val = loadData(name)
    idx = loadData('id')
    c_w_l = loadData(conc)
    shot = loadData('shot')

    #c_w_l must have a value below 1e-2 and nonzero
    good = scipy.logical_and(c_w_l > 0, c_w_l < 1e-2)

    idx = idx[good]
    val = val[good]
    shot = shot[good]
    c_w_l = c_w_l[good]

    #val7 needs to be not at peak value
    good = val < 6e4

    idx = idx[good]
    val = val[good]
    shot = shot[good]
    c_w_l = c_w_l[good]


    val[val < 1.] = 0.

    #normalize to time
    #all valid data with the new computer has 8ms exposure time
    #older shots had 5ms exposure times

    val[shot <= 32858] /= 5e-3
    val[shot > 32858] /= 8e-3

    if nozero:
        good = val != 0
        idx = idx[good]
        val = val[good]
        c_w_l = c_w_l[good]

    
    return val, c_w_l, idx

def conditionVal2(name='val7', nozero=False):
    #this is a program which extracts the useful data from the CSXR data

    val = loadData(name)
    idx = loadData('id')
    c_w_l = loadData('c_w_l')
    c_w_qc = loadData('c_w')
    shot = loadData('shot')

    #c_w_l must have a value below 1e-2 and nonzero
    good = scipy.logical_and(c_w_l > 0, c_w_l < 1e-2)

    #use c_w_qc when c_w_l is bad (and also apply same limitations on concentration value
    good2 = scipy.logical_and(scipy.logical_and(c_w_qc > 0, c_w_qc < 1e-2), c_w_l < 1e-2) #no funny business

    goodtot = scipy.logical_or(good,good2)
    goodmod = scipy.logical_and(scipy.logical_not(good), good2)
    
    idx = idx[goodtot]
    val = val[goodtot]
    shot = shot[goodtot]
    c_w_l[goodmod] = c_w_qc[goodmod]
    c_w_l = c_w_l[goodtot]

    #val7 needs to be not at peak value
    good = val < 6e4

    idx = idx[good]
    val = val[good]
    shot = shot[good]
    c_w_l = c_w_l[good]


    val[val < 1.] = 0.

    #normalize to time
    #all valid data with the new computer has 8ms exposure time
    #older shots had 5ms exposure times

    val[shot <= 32858] /= 5e-3
    val[shot > 32858] /= 8e-3

    if nozero:
        good = val != 0
        idx = idx[good]
        val = val[good]
        c_w_l = c_w_l[good]

    
    return val, c_w_l, idx


def conditionVal3(name1='val1', name2='val2', nozero=False):
    #this is a program which extracts the useful data from the CSXR data

    val = loadData(name1)
    val2 = loadData(name2)
    idx = loadData('id')
    c_w_l = loadData('c_w_l')
    shot = loadData('shot')

    #c_w_l must have a value below 1e-2 and nonzero
    good = scipy.logical_and(c_w_l > 0, c_w_l < 1e-2)

    idx = idx[good]
    val = val[good]
    val2 = val2[good]
    shot = shot[good]
    c_w_l = c_w_l[good]

    #val7 needs to be not at peak value
    good = scipy.logical_and(val < 6e4,val2 < 6e4)

    idx = idx[good]
    val = val[good] + val2[good]
    shot = shot[good]
    c_w_l = c_w_l[good]


    val[val < 1.] = 0.

    #normalize to time
    #all valid data with the new computer has 8ms exposure time
    #older shots had 5ms exposure times

    val[shot <= 32858] /= 5e-3
    val[shot > 32858] /= 8e-3

    if nozero:
        good = val != 0
        idx = idx[good]
        val = val[good]
        c_w_l = c_w_l[good]

    
    return val, c_w_l, idx



def conditionVal4(name='val7', nozero=False, conc='c_w_l'):
    #this is a program which extracts the useful data from the CSXR data

    val = loadData(name)
    idx = loadData('id')
    c_w_l = loadData(conc)
    shot = loadData('shot')

    val[val < 1.] = 0.

    #c_w_l must have a value below 1e-2 and nonzero
    good = scipy.logical_and(c_w_l > 0, c_w_l < 1e-2)


    #val7 needs to be not at peak value
    good = scipy.logical_and(val < 6e4, good)

    #normalize to time
    #all valid data with the new computer has 8ms exposure time
    #older shots had 5ms exposure times

    val[shot <= 32858] /= 5e-3
    val[shot > 32858] /= 8e-3

    if nozero:
        good = scipy.logical_and(val != 0, good)
        


    idx = idx[good]
    val = val[good]
    c_w_l = c_w_l[good]
    shot = shot[good]
    

    
    return val, c_w_l, idx, good

def plottimes():
    time = loadData('time')
    shot = loadData('shot')
    
    uShot = scipy.unique(shot)
    output = scipy.zeros(uShot.shape)
    
    for i in xrange(len(uShot)):
        temp = time[shot == uShot[i]]
        output[i] = scipy.mean(temp[1:]-temp[:-1])


    plt.plot(uShot,output,'.')
    plt.show()


def loadM(idx, database='SIFweights.db'):
    a = sqlite3.connect(database)
    b = a.cursor()
    temp = scipy.squeeze(b.execute('SELECT * FROM shots2016').fetchall())
    order = scipy.searchsorted(temp[:,0],idx)
    return temp[order,1:]


def fitDataLog(name,nozero=True):

    val, c_w_l, idx = conditionVal(name=name, nozero=nozero)

    print('loadM')
    M = loadM(idx)
    
    print('modify M')
    newM = assembleM(M)


    tikh = sklearn.linear_model.RidgeCV()
    
    print('fit Ridge with cross-validation')

    p = tikh.fit(scipy.log(newM/(4*scipy.pi)),scipy.log(val)) #do it in log-log space where I can reasonably assume it is a parabola
    
    print(p.coef_[0],p.coef_[1],p.coef_[2])
    return tikh


def fitDataBFGS(name, init=None, nozero=True): #init is the three initial values of the gaussian needed to fit the data

    val, c_w_l, idx = conditionVal(name=name, nozero=nozero)

    if init is None:
        init = fitPuetti(35)

    print('loadM')
    M = loadM(idx)/1e29 #the M values are 1e29 too high to start, MAKE SURE TO NOT FORGET THIS
                      
    Te = scipy.io.readsav('W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave')['en']
    y = time.time()
    output = scipy.optimize.minimize(objective,
                                     init,
                                     args=(val, Te, c_w_l, M),
                                     bounds=scipy.array(([1e-10,scipy.inf],[1e-10,scipy.inf],[1e-10,scipy.inf])))
    print(time.time()-y)
    return output



def fitPuetti(inp,dataset = 'W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave', lim=1e-2):
    data = scipy.io.readsav(dataset)

    Te = data['en']
    lnTe = scipy.log(Te)
    abund = data['abundance'][:,inp]

    #plt.loglog(Te, abund, lw=2, color='#003153')

    good = abund > lim
    temp = scipy.polyfit(lnTe[good],scipy.log(abund[good]),2)

    #plt.loglog(Te,scipy.exp(temp[0]*pow(lnTe,2)+temp[1]*lnTe+temp[2]),color='#228b22',lw=2)
    #plt.axvline(scipy.exp(-1*temp[1]/(2*temp[0])),lw=2,linestyle='--')
    #plt.axhline(scipy.exp(temp[2]-pow(temp[1]/2.,2)/temp[0]),lw=2,linestyle='--')
    #plt.axvline(Te[abund.argmax()],lw=2,linestyle='--',color='#080808')

    p = scipy.array((scipy.exp(temp[2]-pow((temp[1]+1.)/2.,2)/temp[0]),-1*(temp[1]+1)/(2*temp[0]),1./scipy.sqrt(-1*temp[0])))
    #print(scipy.exp(temp[2]-pow(temp[1]/2.,2)/temp[0]),scipy.exp(-1*(temp[1])/(2*temp[0])))
    #print(inp,scipy.exp(-1*temp[1]/(2*temp[0]))-Te[abund.argmax()],Te[abund.argmax()])
    #plt.loglog(Te,lognorm(Te,p),lw=2,color='#0A2C2C',linestyle=':')#'#0A2C2C')
    return p

                               
def params(p):
    print(scipy.exp(scipy.log(p[0])-p[1]+.25*pow(p[2],2)),scipy.exp(p[1]-pow(p[2],2)/2.))


def HighTe(M, dataset = 'W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave'):
    Te = scipy.io.readsav(dataset)['en']
    output = scipy.zeros((len(M),))
    for i in xrange(len(M)):
        try:
            output[i] = Te[ M[i,:] != 0][-1] #assumes positively increasing ordering
        except IndexError:
            output[i] = Te[0] # THIS IS TO COVER A PROBLEM IN THE DATABASE, I need to go back and work on this.
            
    return output


def visualize(M, name, nozero=False):

    val, c_w_l, idx = conditionVal(name=name, nozero=nozero)
              
    data = scipy.io.readsav('W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave')
    y = time.time()

    maxT = HighTe(M)
    plt.loglog(maxT,val/c_w_l/scipy.sum(M,axis=1)*1e30,'.',alpha=.01)
    plt.loglog(data['en'],data['abundance'][:,46])
    #plt.show()

def visualize2(M, name, nozero=False, k=2e29, useall=False, lim=1e-2, conc='c_w_l'):

    if useall:
        val, c_w_l, idx = conditionVal2(name=name, nozero=nozero, conc=conc)
    else:
        val, c_w_l, idx = conditionVal(name=name, nozero=nozero, conc=conc)
              
    data = scipy.io.readsav('W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave')
    y = time.time()

    maxT = HighTe(M)
    #plt.loglog(maxT,val/c_w_l/scipy.sum(M,axis=1)*1e30,'.',alpha=.01)
    xax = scipy.logspace(2,5,201)
    yax = scipy.logspace(-5,1,101)
    xtemp = xax[1:]/2.+xax[:-1]/2.
    ytemp = yax[1:]/2.+yax[:-1]/2.

    Y,X = scipy.meshgrid(xtemp, ytemp)
    histdata, xed, yed = scipy.histogram2d(maxT,val/c_w_l/scipy.sum(M,axis=1)*k,bins=[xax,yax])
    extent = [xed[0], xed[-1], yed[0], yed[-1]]
    plt.pcolormesh(xax,yax,histdata.T,norm=LogNorm(), cmap='viridis', rasterized=True)
    plt.gca().set_yscale('log')
    plt.gca().set_xscale('log')
    plt.gca().set_ylim(1e-4,1e1)
    plt.gca().set_xlim(1e3,2e4)
    


    #plt.loglog(data['en'],data['abundance'][:,46],lw=2,color='w')    
    #plt.loglog(data['en'],data['abundance'][:,46],lw=2,color='b',alpha=.4,linestyle='--')
    #plt.loglog(data['en'],scipy.cumsum(data['abundance'][:,46])/scipy.sum(data['abundance'][:,46]),lw=2,color='w')
    #plt.loglog(data['en'],scipy.cumsum(data['abundance'][:,46])/scipy.sum(data['abundance'][:,46]),lw=2,color='b',alpha=.4,linestyle='--')
    #plt.loglog(data['en'],data['abundance'][:,45],lw=2,color='w')
    #plt.loglog(data['en'],data['abundance'][:,45],lw=2,color='g',alpha=.4,linestyle='--')
    #plt.loglog(data['en'],scipy.cumsum(data['abundance'][:,45])/scipy.sum(data['abundance'][:,45]),lw=2,color='w')
    #plt.loglog(data['en'],scipy.cumsum(data['abundance'][:,45])/scipy.sum(data['abundance'][:,45]),lw=2,color='g',alpha=.4,linestyle='--')
    #plt.loglog(data['en'],data['abundance'][:,44],lw=2,color='w')
    #plt.loglog(data['en'],data['abundance'][:,44],lw=2,color='r',alpha=.4,linestyle='--')
    #plt.loglog(data['en'],scipy.cumsum(data['abundance'][:,44])/scipy.sum(data['abundance'][:,44]),lw=2,color='w')
    #plt.loglog(data['en'],scipy.cumsum(data['abundance'][:,44])/scipy.sum(data['abundance'][:,44]),lw=2,color='r',alpha=.4,linestyle='--')

    if True:
        alpha=.8
        idx = scipy.logical_and(data['en'] > 1e3, data['en'] < 5e4)
        output = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        for i in temp:
            #output[i] = scipy.sum(testM.normMexp(scipy.log(data['en']),  scipy.log(6.5e2), scipy.log(data['en'][i]), .08, .8e-1)*data['abundance'][:,46])
            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,46])

        #plt.loglog(data['en'],output/output.max(),lw=2,color='w',alpha=alpha,label='_nolegend_')
        plt.loglog(data['en'],output/output.max(),lw=3,color='crimson',alpha=alpha,linestyle='--',label='W$^{46+}$')   


        output = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        for i in temp:
            
            #output[i] = scipy.sum(testM.normMexp(scipy.log(data['en']),  scipy.log(6.5e2), scipy.log(data['en'][i]), .08, .8e-1)*data['abundance'][:,45])
            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,45])

        #plt.loglog(data['en'],output/output.max(),lw=2,color='w',alpha=alpha,label='_nolegend_')
        plt.loglog(data['en'],output/output.max(),lw=3,color='magenta',alpha=alpha,linestyle='--',label='W$^{45+}$')


        output = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        for i in temp:

            #output[i] = scipy.sum(testM.normMexp(scipy.log(data['en']),  scipy.log(6.5e2), scipy.log(data['en'][i]), .08, .8e-1)*data['abundance'][:,44])
            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,44])

        #plt.loglog(data['en'],output/output.max(),lw=2,color='w',alpha=alpha,label='_nolegend_')
        plt.loglog(data['en'],output/output.max(),lw=3,color='cyan',alpha=alpha,linestyle='--',label='W$^{44+}$')

        plt.xlabel(r'$T_e$ [eV]')
        plt.ylabel(r'const$\cdot I_{CSXR} / c_W \sum M$')

    colorbar = plt.colorbar()
    colorbar.ax.set_ylabel('datapoints in bin')
    plt.legend(loc=4,fontsize=20)
    plt.subplots_adjust(bottom=.12,right=1.)

    #plt.show()

def visualize3(M,idx1,idx2,en=5e2,rat=.125):

    data = scipy.io.readsav('W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave')
    plt.semilogx(data['en'],M[idx1:idx2,:].T/scipy.sum(M[idx1:idx2,:],axis=1),alpha=.01,color='b')
    #plt.semilogx(data['en'],testM.normMn2(10, scipy.log(data['en']),  scipy.log(6.5e2), scipy.log(3.2e3), .125),color='r',lw=2,linestyle='--')
    plt.semilogx(data['en'],testM.normMexp(scipy.log(data['en']),  scipy.log(6.5e2), scipy.log(3.2e3), .08, .8e-1),color='r',lw=2,linestyle='--')
    plt.xlim(1e2,1e4)
    plt.ylim(0,.06)
    plt.ylabel('Normalized M')
    plt.xlabel('$T_e$ [eV]')
    plt.subplots_adjust(bottom=.12)
    plt.show()


def visualize4(M, nozero=False, k = 2e29):

    val, c_w_l, idx = conditionVal3(nozero=nozero)


    data = scipy.io.readsav('W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave')
    y = time.time()

    maxT = HighTe(M)
    #plt.loglog(maxT,val/c_w_l/scipy.sum(M,axis=1)*1e30,'.',alpha=.01)
    xax = scipy.logspace(2,5,201)
    yax = scipy.logspace(-5,1,101)
    xtemp = xax[1:]/2.+xax[:-1]/2.
    ytemp = yax[1:]/2.+yax[:-1]/2.

    Y,X = scipy.meshgrid(xtemp, ytemp)
    histdata, xed, yed = scipy.histogram2d(maxT,val/c_w_l/scipy.sum(M,axis=1)*k,bins=[xax,yax])
    extent = [xed[0], xed[-1], yed[0], yed[-1]]
    plt.pcolormesh(xtemp,ytemp,histdata.T,norm=LogNorm(), cmap='viridis', rasterized=True)
    plt.gca().set_yscale('log')
    plt.gca().set_xscale('log')
    plt.gca().set_ylim(1e-4,1e1)
    plt.gca().set_xlim(1e3,2e4)
    


    #plt.loglog(data['en'],data['abundance'][:,46],lw=2,color='w')    
    #plt.loglog(data['en'],data['abundance'][:,46],lw=2,color='b',alpha=.4,linestyle='--')
    #plt.loglog(data['en'],scipy.cumsum(data['abundance'][:,46])/scipy.sum(data['abundance'][:,46]),lw=2,color='w')
    #plt.loglog(data['en'],scipy.cumsum(data['abundance'][:,46])/scipy.sum(data['abundance'][:,46]),lw=2,color='b',alpha=.4,linestyle='--')
    #plt.loglog(data['en'],data['abundance'][:,45],lw=2,color='w')
    #plt.loglog(data['en'],data['abundance'][:,45],lw=2,color='g',alpha=.4,linestyle='--')
    #plt.loglog(data['en'],scipy.cumsum(data['abundance'][:,45])/scipy.sum(data['abundance'][:,45]),lw=2,color='w')
    #plt.loglog(data['en'],scipy.cumsum(data['abundance'][:,45])/scipy.sum(data['abundance'][:,45]),lw=2,color='g',alpha=.4,linestyle='--')
    #plt.loglog(data['en'],data['abundance'][:,44],lw=2,color='w')
    #plt.loglog(data['en'],data['abundance'][:,44],lw=2,color='r',alpha=.4,linestyle='--')
    #plt.loglog(data['en'],scipy.cumsum(data['abundance'][:,44])/scipy.sum(data['abundance'][:,44]),lw=2,color='w')
    #plt.loglog(data['en'],scipy.cumsum(data['abundance'][:,44])/scipy.sum(data['abundance'][:,44]),lw=2,color='r',alpha=.4,linestyle='--')

    if True:
        alpha=.8
        idx = scipy.logical_and(data['en'] > 1e3, data['en'] < 5e4)
        output = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        for i in temp:
            #output[i] = scipy.sum(testM.normMexp(scipy.log(data['en']),  scipy.log(6.5e2), scipy.log(data['en'][i]), .08, .8e-1)*data['abundance'][:,46])
            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,46])

        #plt.loglog(data['en'],output/output.max(),lw=2,color='w',alpha=alpha,label='_nolegend_')
        plt.loglog(data['en'],output/output.max(),lw=3,color='crimson',alpha=alpha,linestyle='--',label='W$^{46+}$')   


        output = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        for i in temp:
            
            #output[i] = scipy.sum(testM.normMexp(scipy.log(data['en']),  scipy.log(6.5e2), scipy.log(data['en'][i]), .08, .8e-1)*data['abundance'][:,45])
            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,45])

        #plt.loglog(data['en'],output/output.max(),lw=2,color='w',alpha=alpha,label='_nolegend_')
        plt.loglog(data['en'],output/output.max(),lw=3,color='magenta',alpha=alpha,linestyle='--',label='W$^{45+}$')


        output = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        for i in temp:

            #output[i] = scipy.sum(testM.normMexp(scipy.log(data['en']),  scipy.log(6.5e2), scipy.log(data['en'][i]), .08, .8e-1)*data['abundance'][:,44])
            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,44])

        #plt.loglog(data['en'],output/output.max(),lw=2,color='w',alpha=alpha,label='_nolegend_')
        plt.loglog(data['en'],output/output.max(),lw=3,color='cyan',alpha=alpha,linestyle='--',label='W$^{44+}$')
        plt.xlabel(r'$T_e$ [eV]')
        plt.ylabel(r'const$\cdot I_{CSXR} / c_W \sum M$')

    colorbar = plt.colorbar()
    colorbar.ax.set_ylabel('datapoints in bin')
    plt.legend(loc=4,fontsize=20)
    plt.subplots_adjust(bottom=.12,right=1.)

    #plt.show()

def visualize5(M, name, nozero=False,k = 2e29,useall=False,lim=1e-2):

    if useall:
        val, c_w_l, idx = conditionVal2(name=name, nozero=nozero)
    else:
        val, c_w_l, idx = conditionVal(name=name, nozero=nozero)
              
    data = scipy.io.readsav('W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave')
    y = time.time()

    maxT = HighTe(M)
    #plt.loglog(maxT,val/c_w_l/scipy.sum(M,axis=1)*1e30,'.',alpha=.01)
    xax = scipy.logspace(2,5,201)
    yax = scipy.logspace(-5,1,101)
    xtemp = xax[1:]/2.+xax[:-1]/2.
    ytemp = yax[1:]/2.+yax[:-1]/2.

    Y,X = scipy.meshgrid(xtemp, ytemp)
    histdata, xed, yed = scipy.histogram2d(maxT,val/c_w_l/scipy.sum(M,axis=1)*k,bins=[xax,yax])
    extent = [xed[0], xed[-1], yed[0], yed[-1]]
    plt.pcolormesh(xtemp,ytemp,histdata.T,norm=LogNorm(), cmap='viridis', rasterized=True)
    plt.gca().set_yscale('log')
    plt.gca().set_xscale('log')
    plt.gca().set_ylim(1e-4,1e1)
    plt.gca().set_xlim(1e3,2e4)
    


    #plt.loglog(data['en'],data['abundance'][:,46],lw=2,color='w')    
    #plt.loglog(data['en'],data['abundance'][:,46],lw=2,color='b',alpha=.4,linestyle='--')
    #plt.loglog(data['en'],scipy.cumsum(data['abundance'][:,46])/scipy.sum(data['abundance'][:,46]),lw=2,color='w')
    #plt.loglog(data['en'],scipy.cumsum(data['abundance'][:,46])/scipy.sum(data['abundance'][:,46]),lw=2,color='b',alpha=.4,linestyle='--')
    #plt.loglog(data['en'],data['abundance'][:,45],lw=2,color='w')
    #plt.loglog(data['en'],data['abundance'][:,45],lw=2,color='g',alpha=.4,linestyle='--')
    #plt.loglog(data['en'],scipy.cumsum(data['abundance'][:,45])/scipy.sum(data['abundance'][:,45]),lw=2,color='w')
    #plt.loglog(data['en'],scipy.cumsum(data['abundance'][:,45])/scipy.sum(data['abundance'][:,45]),lw=2,color='g',alpha=.4,linestyle='--')
    #plt.loglog(data['en'],data['abundance'][:,44],lw=2,color='w')
    #plt.loglog(data['en'],data['abundance'][:,44],lw=2,color='r',alpha=.4,linestyle='--')
    #plt.loglog(data['en'],scipy.cumsum(data['abundance'][:,44])/scipy.sum(data['abundance'][:,44]),lw=2,color='w')
    #plt.loglog(data['en'],scipy.cumsum(data['abundance'][:,44])/scipy.sum(data['abundance'][:,44]),lw=2,color='r',alpha=.4,linestyle='--')

    if True:
        alpha=.8
        idx = scipy.logical_and(data['en'] > 1e3, data['en'] < 5e4)
        output = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        for i in temp:
            #output[i] = scipy.sum(testM.normMexp(scipy.log(data['en']),  scipy.log(6.5e2), scipy.log(data['en'][i]), .08, .8e-1)*data['abundance'][:,46])
            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,46])

        #plt.loglog(data['en'],output/output.max(),lw=2,color='w',alpha=alpha,label='_nolegend_')
        plt.loglog(data['en'],output/output.max(),lw=3,color='crimson',alpha=alpha,linestyle='--',label='W$^{46+}$')   


        output = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        for i in temp:
            
            #output[i] = scipy.sum(testM.normMexp(scipy.log(data['en']),  scipy.log(6.5e2), scipy.log(data['en'][i]), .08, .8e-1)*data['abundance'][:,45])
            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,45])

        #plt.loglog(data['en'],output/output.max(),lw=2,color='w',alpha=alpha,label='_nolegend_')
        plt.loglog(data['en'],output/output.max(),lw=3,color='magenta',alpha=alpha,linestyle='--',label='W$^{45+}$')
       

        output = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        for i in temp:


            para = fitPuetti(46,lim=lim)
            #output[i] = scipy.sum(testM.normMexp(scipy.log(data['en']),  scipy.log(6.5e2), scipy.log(data['en'][i]), .08, .8e-1)*data['abundance'][:,44])
            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*(lognorm(data['en'],para)))

        #plt.loglog(data['en'],output/output.max(),lw=2,color='w',alpha=alpha,label='_nolegend_')
        plt.loglog(data['en'],output/output.max(),lw=3,color='crimson',alpha=alpha,linestyle='-',label='$^*$W$^{46+}$')

        output = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        for i in temp:


            para = fitPuetti(45,lim=lim)
            #output[i] = scipy.sum(testM.normMexp(scipy.log(data['en']),  scipy.log(6.5e2), scipy.log(data['en'][i]), .08, .8e-1)*data['abundance'][:,44])
            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*(lognorm(data['en'],para)))

        #plt.loglog(data['en'],output/output.max(),lw=2,color='w',alpha=alpha,label='_nolegend_')
        plt.loglog(data['en'],output/output.max(),lw=3,color='magenta',alpha=alpha,linestyle='-',label='$^*$W$^{45+}$')


        plt.xlabel(r'$T_e$ [eV]')
        plt.ylabel(r'const$\cdot I_{CSXR} / c_W \sum M$')

    colorbar = plt.colorbar()
    colorbar.ax.set_ylabel('datapoints in bin')
    plt.legend(loc=4,fontsize=20)
    plt.subplots_adjust(bottom=.12,right=1.)

    #plt.show()

def fitPuettiPlot(inp=46,inp2=45,dataset = 'W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave', lim=1e-2,loc=8):
    data = scipy.io.readsav(dataset)

    Te = data['en']
    lnTe = scipy.log(Te)
    abund = data['abundance'][:,inp]

    #plt.loglog(Te, abund, lw=2, color='#003153')

    good = abund > lim
    temp = scipy.polyfit(lnTe[good],scipy.log(abund[good]),2)
    plt.loglog(Te,abund,color='crimson',linestyle='--',lw=2,label=r'W$^{46+}$ data')

    plt.loglog(Te,data['abundance'][:,inp2],color='magenta',lw=2,linestyle='--',label=r'W$^{45+}$ data')

    #plt.loglog(Te,scipy.exp(temp[0]*pow(lnTe,2)+temp[1]*lnTe+temp[2]),color='#228b22',lw=2,label=r'P\"utterich data')
    #plt.axvline(scipy.exp(-1*temp[1]/(2*temp[0])),lw=2,linestyle='--')
    #plt.axhline(scipy.exp(temp[2]-pow(temp[1]/2.,2)/temp[0]),lw=2,linestyle='--')
    #plt.axvline(Te[abund.argmax()],lw=2,linestyle='--',color='#080808')

    p = scipy.array((scipy.exp(temp[2]-pow((temp[1]+1.)/2.,2)/temp[0]),-1*(temp[1]+1)/(2*temp[0]),1./scipy.sqrt(-1*temp[0])))
    #print(scipy.exp(temp[2]-pow(temp[1]/2.,2)/temp[0]),scipy.exp(-1*(temp[1])/(2*temp[0])))
    #print(inp,scipy.exp(-1*temp[1]/(2*temp[0]))-Te[abund.argmax()],Te[abund.argmax()])
    plt.loglog(Te,lognorm(Te,p),lw=2,color='crimson',linestyle='-',label='W$^{46+}$ fit')#'#0A2C2C')

    abund = data['abundance'][:,inp2]

    #plt.loglog(Te, abund, lw=2, color='#003153')

    good = abund > lim
    temp = scipy.polyfit(lnTe[good],scipy.log(abund[good]),2)
    #plt.loglog(Te,abund,color='crimson',lw=2,linestyle='--',label=r'W$^{46+}$ data')


    #plt.loglog(Te,scipy.exp(temp[0]*pow(lnTe,2)+temp[1]*lnTe+temp[2]),color='#228b22',lw=2,label=r'P\"utterich data')
    #plt.axvline(scipy.exp(-1*temp[1]/(2*temp[0])),lw=2,linestyle='--')
    #plt.axhline(scipy.exp(temp[2]-pow(temp[1]/2.,2)/temp[0]),lw=2,linestyle='--')
    #plt.axvline(Te[abund.argmax()],lw=2,linestyle='--',color='#080808')

    p = scipy.array((scipy.exp(temp[2]-pow((temp[1]+1.)/2.,2)/temp[0]),-1*(temp[1]+1)/(2*temp[0]),1./scipy.sqrt(-1*temp[0])))
    #print(scipy.exp(temp[2]-pow(temp[1]/2.,2)/temp[0]),scipy.exp(-1*(temp[1])/(2*temp[0])))
    #print(inp,scipy.exp(-1*temp[1]/(2*temp[0]))-Te[abund.argmax()],Te[abund.argmax()])
    plt.loglog(Te,lognorm(Te,p),lw=2,color='magenta',linestyle='-',label='W$^{45+}$ fit')#'#0A2C2C')

    plt.fill_between([1e3,2e4],[lim,lim],[1.,1.],color='#DCDCDC',zorder=0)

    plt.xlabel(r'$T_e$ [eV]')
    plt.ylabel(r'fractional abundance')
    plt.gca().set_ylim([1e-5,1e0])
    plt.gca().set_xlim(1e3,2e4)
    plt.legend(loc=loc,fontsize=20)
    plt.subplots_adjust(bottom=.12)
    #return p

def visualize6(M, name, nozero=False, k=2., useall=False, lim=1e-2, conc='c_w_l',W=46,color='crimson'):
    """ visualize variation in the the profile shape"""

    if useall:
        val, c_w_l, idx = conditionVal2(name=name, nozero=nozero, conc=conc)
    else:
        val, c_w_l, idx = conditionVal(name=name, nozero=nozero, conc=conc)
              
    data = scipy.io.readsav('W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave')
    y = time.time()

    maxT = HighTe(M)
    output = scipy.zeros(maxT.shape)
    print(output.shape)
    Msum = scipy.sum(M,axis=1)
    maxval = data['abundance'][:,W].max()
    for i in xrange(len(output)):
        output[i] = scipy.sum(M[i]*data['abundance'][:,W])/Msum[i]/maxval*k #should be same size as maxT


    #plt.loglog(maxT,val/c_w_l/scipy.sum(M,axis=1)*1e30,'.',alpha=.01)
    xax = scipy.logspace(2,5,201)
    yax = scipy.logspace(-5,1,101)
    xtemp = xax[1:]/2.+xax[:-1]/2.
    ytemp = yax[1:]/2.+yax[:-1]/2.

    Y,X = scipy.meshgrid(xtemp, ytemp)
    histdata, xed, yed = scipy.histogram2d(maxT,output,bins=[xax,yax])
    extent = [xed[0], xed[-1], yed[0], yed[-1]]
    colormesh = plt.pcolormesh(xax,yax,histdata.T,norm=LogNorm(), cmap='viridis', rasterized=True)
    plt.gca().set_yscale('log')
    plt.gca().set_xscale('log')
    plt.gca().set_ylim(1e-4,1e1)
    plt.gca().set_xlim(1e3,2e4)
    
    if True:
        alpha=.8
        idx = scipy.logical_and(data['en'] > 1e3, data['en'] < 5e4)
        output = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        for i in temp:
            #output[i] = scipy.sum(testM.normMexp(scipy.log(data['en']),  scipy.log(6.5e2), scipy.log(data['en'][i]), .08, .8e-1)*data['abundance'][:,46])
            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,W])

        #plt.loglog(data['en'],output/output.max(),lw=2,color='w',alpha=alpha,label='_nolegend_')
        plt.loglog(data['en'],output/output.max(),lw=3,color=color,alpha=alpha,linestyle='--',label=r'W$^{'+str(W)+'+}$')   

        plt.xlabel(r'$T_e$ [eV]')
        #plt.ylabel(r'const $\cdot \sum (M \cdot f_Z(T_e)) / \sum M$')

    #colorbar = plt.colorbar()
    #colorbar.ax.set_ylabel('datapoints in bin')
    plt.legend(loc=4,fontsize=20)
    plt.subplots_adjust(wspace=0.,bottom=.12,right=1.)

    return colormesh
    #plt.show()


def plotPEC(loc='data_to_Ian.txt'):
    nov = scipy.array([1,1,1,1,1])
    colors=scipy.array(['yellow','cyan','magenta','crimson','#800080']) #should've used a dict
    nums = scipy.array([43,44,45,46,47])
    te = scipy.io.readsav('W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave')['en']

    inp = adsre.lineInfo(loc)

    for i in xrange(len(inp.z)):
        idx = nums == inp.charge[i]
        print(nums[idx][0])
        if nov[idx][0] == 1:
            plt.loglog(te,inp(te,i)/inp(te,i).max(),linewidth=2,color=colors[idx][0],label='W$^{'+str(nums[idx][0])+'+}$')
            nov[idx] = 0
            print(nov)
        else:
            plt.loglog(te,inp(te,i),linewidth=2,color=colors[idx][0],label='_nolegend_')

    plt.gca().set_ylim(1e-1,1e0)
    plt.gca().set_xlim(1e3,2e4)
    plt.xlabel('$T_e$ [eV]')
    plt.ylabel('normalized PEC($T_e$)')
    plt.legend(loc=4,fontsize=20)
    plt.subplots_adjust(bottom=.12)
    plt.show()
   

def visualize2PEC(M, name, nozero=False, k=2.5e29, useall=False, lim=1e-2, conc='c_w_l',loc='data_to_Ian.txt'):
    """ log plot of the data comparison with and without PEC data"""
    if useall:
        val, c_w_l, idx = conditionVal2(name=name, nozero=nozero, conc=conc)
    else:
        val, c_w_l, idx = conditionVal(name=name, nozero=nozero, conc=conc)
              
    data = scipy.io.readsav('W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave')
    y = time.time()

    maxT = HighTe(M)
    #plt.loglog(maxT,val/c_w_l/scipy.sum(M,axis=1)*1e30,'.',alpha=.01)
    xax = scipy.logspace(2,5,201)
    yax = scipy.logspace(-5,1,101)
    xtemp = xax[1:]/2.+xax[:-1]/2.
    ytemp = yax[1:]/2.+yax[:-1]/2.

    Y,X = scipy.meshgrid(xtemp, ytemp)
    histdata, xed, yed = scipy.histogram2d(maxT,val/c_w_l/scipy.sum(M,axis=1)*k,bins=[xax,yax])
    extent = [xed[0], xed[-1], yed[0], yed[-1]]
    plt.pcolormesh(xax,yax,histdata.T, cmap='viridis', rasterized=True, norm=LogNorm())
    plt.gca().set_yscale('log')
    plt.gca().set_xscale('log')
    plt.gca().set_ylim(1e-4,1e1)
    plt.gca().set_xlim(1e3,2e4)
   
    cmap = plt.cm.get_cmap('viridis')
    cmap.set_under('white')
 
    inp = adsre.lineInfo(loc)
    #plt.plot([],[],color='k',lw=2,label=r'with PEC') 


    if True:
        alpha=.8
        idx = scipy.logical_and(data['en'] > 1e3, data['en'] < 5e4)
        output = scipy.zeros(data['en'].shape)
        output2 = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        for i in temp:
            PEC = inp(data['en'], 0)
            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,46]*PEC)
            output2[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,46])

        plt.loglog(data['en'],output2/output2.max(),lw=3,color='crimson',alpha=alpha,linestyle='--',label=r'W$^{46+}$')

        plt.loglog(data['en'],output/output.max(),lw=3,color='crimson',alpha=alpha,linestyle='-',label=r'_nolegend_')   
        output = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        for i in temp:
            PEC = inp(data['en'], 3)
            
            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,45]*PEC)
            output2[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,45])


        plt.loglog(data['en'],output2/output2.max(),lw=3,color='magenta',alpha=alpha,linestyle='--',label=r'W$^{45+}$')
        plt.loglog(data['en'],output/output.max(),lw=3,color='magenta',alpha=alpha,linestyle='-',label=r'_nolegend')

        output = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        for i in temp:
            PEC = inp(data['en'], 6)

            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,44]*PEC)
            output2[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,44])

        plt.loglog(data['en'],output2/output2.max(),lw=3,color='cyan',alpha=alpha,linestyle='--',label=r'W$^{44+}$')
        plt.loglog(data['en'],output/output.max(),lw=3,color='cyan',alpha=alpha,linestyle='-',label=r'_nolegend_')

        plt.xlabel(r'$T_e$ [eV]')
        plt.ylabel(r'const$\cdot I_{CSXR} / c_W \sum M$')

    colorbar = plt.colorbar()
    colorbar.ax.set_ylabel('datapoints in bin')
    leg = plt.legend(loc=4,fontsize=20,title=r'\underline{\emph{solid} - \emph{with PEC}}')
    plt.setp(leg.get_title(),fontsize=14)
    plt.subplots_adjust(bottom=.12,right=1.)

    #plt.show()

def visualize2PEC2(M, name, nozero=False, k=2.5e29, useall=False, lim=1e-2, conc='c_w_l',loc='data_to_Ian.txt'):
    """ log plot of the data with only PEC data"""
    if useall:
        val, c_w_l, idx = conditionVal2(name=name, nozero=nozero, conc=conc)
    else:
        val, c_w_l, idx = conditionVal(name=name, nozero=nozero, conc=conc)
              
    data = scipy.io.readsav('W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave')
    y = time.time()

    maxT = HighTe(M)
    #plt.loglog(maxT,val/c_w_l/scipy.sum(M,axis=1)*1e30,'.',alpha=.01)
    xax = scipy.logspace(2,5,201)
    yax = scipy.logspace(-5,1,101)
    xtemp = xax[1:]/2.+xax[:-1]/2.
    ytemp = yax[1:]/2.+yax[:-1]/2.

    Y,X = scipy.meshgrid(xtemp, ytemp)
    histdata, xed, yed = scipy.histogram2d(maxT,val/c_w_l/scipy.sum(M,axis=1)*k,bins=[xax,yax])
    extent = [xed[0], xed[-1], yed[0], yed[-1]]
    plt.pcolormesh(xax,yax,histdata.T, cmap='viridis', rasterized=True, norm=LogNorm())
    plt.gca().set_yscale('log')
    plt.gca().set_xscale('log')
    plt.gca().set_ylim(1e-4,1e1)
    plt.gca().set_xlim(1e3,2e4)
   
    cmap = plt.cm.get_cmap('viridis')
    cmap.set_under('white')
 
    inp = adsre.lineInfo(loc)
    #plt.plot([],[],color='k',lw=2,label=r'with PEC') 


    if True:
        alpha=.8
        idx = scipy.logical_and(data['en'] > 1e3, data['en'] < 5e4)
        output = scipy.zeros(data['en'].shape)
        output2 = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        for i in temp:
            PEC = inp(data['en'], 0)
            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,46]*PEC)
            output2[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,46])

        plt.loglog(data['en'],output/output.max(),lw=3,color='crimson',alpha=alpha,linestyle='--',label=r'W$^{46+}$')

        #plt.loglog(data['en'],output/output.max(),lw=3,color='crimson',alpha=alpha,linestyle='-',label=r'_nolegend_')   
        output = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        for i in temp:
            PEC = inp(data['en'], 3)
            
            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,45]*PEC)
            output2[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,45])


        plt.loglog(data['en'],output/output.max(),lw=3,color='magenta',alpha=alpha,linestyle='--',label=r'W$^{45+}$')
        #plt.loglog(data['en'],output/output.max(),lw=3,color='magenta',alpha=alpha,linestyle='-',label=r'_nolegend')

        output = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        for i in temp:
            PEC = inp(data['en'], 6)

            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,44]*PEC)
            output2[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,44])

        plt.loglog(data['en'],output/output.max(),lw=3,color='cyan',alpha=alpha,linestyle='--',label=r'W$^{44+}$')
        #plt.loglog(data['en'],output/output.max(),lw=3,color='cyan',alpha=alpha,linestyle='-',label=r'_nolegend_')

        plt.xlabel(r'$T_e$ [eV]')
        plt.ylabel(r'const$\cdot I_{CSXR} / c_W \sum M$')

    colorbar = plt.colorbar()
    colorbar.ax.set_ylabel('datapoints in bin')
    leg = plt.legend(loc=4,fontsize=20,title=r'\underline{\hspace{1em} \emph{with PEC} \hspace{1em}}')
    plt.setp(leg.get_title(),fontsize=14)
    plt.subplots_adjust(bottom=.12,right=1.)

    #plt.show()

def visualize2PEC3(M, name, nozero=False, k=2.636e29, useall=False, lim=1e-2, conc='c_w_l',loc='data_to_Ian.txt'):
    """ goodness of fit vs charge state fractional abundance with expected PEC """

    if useall:
        val, c_w_l, idx = conditionVal2(name=name, nozero=nozero, conc=conc)
    else:
        val, c_w_l, idx = conditionVal(name=name, nozero=nozero, conc=conc)
              
    data = scipy.io.readsav('W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave')
    y = time.time()

    maxT = HighTe(M)
    #plt.loglog(maxT,val/c_w_l/scipy.sum(M,axis=1)*1e30,'.',alpha=.01)
    xax = scipy.logspace(2,5,201)
    yax = scipy.logspace(-5,1,101)

    expdata = val/c_w_l/scipy.sum(M,axis=1)
    histdata, xed, yed = scipy.histogram2d(maxT,expdata*k,bins=[xax,yax])

    


    #plt.pcolormesh(xax,yax,histdata.T, cmap='viridis', rasterized=True, norm=LogNorm())
    #plt.gca().set_yscale('log')
    #plt.gca().set_xscale('log')
    #plt.gca().set_ylim(1e-4,1e1)
    #plt.gca().set_xlim(1e3,2e4)
   
    #cmap = plt.cm.get_cmap('viridis')
    #cmap.set_under('white')
 
    inp = adsre.lineInfo(loc)
    #plt.plot([],[],color='k',lw=2,label=r'with PEC') 
    plt.figure(figsize=(8,4))

    pecDict = {43:7,44:6,45:3,46:0,47:8} #related to line PEC document
    rsquared = scipy.nan*scipy.zeros((data['abundance'].shape[1]-1,))
    rsquared2 = scipy.nan*scipy.zeros((data['abundance'].shape[1]-1,))
    chargestates = scipy.arange(len(rsquared))+1
    print(len(rsquared),data['abundance'].shape)


    for j in chargestates:
        idx = scipy.logical_and(data['en'] > 1e3, data['en'] < 5e4)
        alpha=.8
        output = scipy.zeros(data['en'].shape)
        output2 = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]



        noPEC = False
        
        try:
            PEC = inp(data['en'], pecDict[j])
            print(j)
        except KeyError:
            PEC = scipy.ones(data['en'].shape)
            noPEC = True

        for i in temp:
            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,j]*PEC)
            output2[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,j])

            #datafitting here
        p = scipy.arange(len(data['en']))[scipy.searchsorted(data['en'][1:]/2.+data['en'][:-1]/2.,maxT)]

        lims = scipy.logical_and(data['en'][p] > 2e3, data['en'][p] < 2e4)
        idx = scipy.logical_and(output[p] != 0, expdata !=0)
        idx2 = scipy.logical_and(output2[p] != 0, expdata !=0)

        idx12 = scipy.logical_and(lims, idx)
        idx22 = scipy.logical_and(lims, idx2)

        differ = scipy.log(output[p][idx12]/output.max())-scipy.log(expdata[idx12])
        differ2 = scipy.log(output2[p][idx22]/output2.max())-scipy.log(expdata[idx22])
        



        #print(scipy.exp(scipy.median(differ)))
        #edges = scipy.linspace(60,80,5e2)
        #histdata = scipy.histogram(differ,bins=edges)
        #plt.semilogx(scipy.exp(edges[1:]),histdata[0])
        #plt.axvline(scipy.exp(scipy.nanmean(differ)),color='g')
        #plt.axvline(scipy.exp(scipy.median(differ)),color='r')
        #print(scipy.var(differ))
        if not noPEC:
            rsquared[j-1]= 1- scipy.var(differ)/scipy.var(scipy.log(expdata[idx12]))       
        rsquared2[j-1]= 1- scipy.var(differ2)/scipy.var(scipy.log(expdata[idx22]))

    plt.plot(chargestates,rsquared,color='#A02C2C',marker='s',markersize=10,lw=0,label=r'PEC$\cdot f_Z$')
    plt.plot(chargestates,rsquared2,color='#228B22',marker='o',markersize=10,lw=0,label=r'$ f_Z$')
    plt.xlim([40,50])
    plt.xticks(scipy.mgrid[40:51:1])
    plt.ylim([0,1])
    #plt.text(41,.6,r'peak $R^2=.514$ \\ $(W^{45+})$')

    plt.xlabel(r'charge state')
    plt.ylabel(r'$R^2$ of $\log$-$\log$ fit')
    leg = plt.legend(numpoints=1,fontsize=18,title=r'\underline{\hspace{.5ex} max$(R^2)=.514$ \hspace{.5ex}}')
    plt.setp(leg.get_title(),fontsize=14)
    plt.subplots_adjust(bottom=.2)
    plt.show()



        #plt.loglog(data['en'],output2/output2.max(),lw=3,color='magenta',alpha=alpha,linestyle='--',label=r'W$^{45+}$')
        #plt.loglog(data['en'],output/output.max(),lw=3,color='magenta',alpha=alpha,linestyle='-',label=r'_nolegend')
        #plt.xlabel(r'$T_e$ [eV]')
        #plt.ylabel(r'const$\cdot I_{CSXR} / c_W \sum M$')

    return output2[p],expdata

def visualize2PEC4(M, name, nozero=False, k=2.636e29, useall=False, lim=1e-2, conc='c_w_l',loc='data_to_Ian.txt'):
    """ this plots the heteroskedasticity of the data (which I need to further investigate """
    if useall:
        val, c_w_l, idx = conditionVal2(name=name, nozero=nozero, conc=conc)
    else:
        val, c_w_l, idx = conditionVal(name=name, nozero=nozero, conc=conc)
              
    data = scipy.io.readsav('W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave')

    maxT = HighTe(M)        
    p = scipy.arange(len(data['en']))[scipy.searchsorted(data['en'][1:]/2.+data['en'][:-1]/2.,maxT)]


    xax = scipy.logspace(2,5,201)
    yax = scipy.logspace(-5,1,101)

    expdata = val/c_w_l/scipy.sum(M,axis=1)

    #plt.gca().set_yscale('log')
    plt.gca().set_xscale('log')
    #plt.gca().set_ylim(1e-4,1e1)
    plt.gca().set_xlim(1e3,2e4)
   
    inp = adsre.lineInfo(loc)

    pecDict = {43:7,44:6,45:3,46:0,47:8} #related to line PEC document
    variances =  scipy.zeros(data['en'].shape)

    if True:

        j = 45
        idx = scipy.logical_and(data['en'] > 1e3, data['en'] < 5e4)
        alpha=.8
        output = scipy.zeros(data['en'].shape)
        output2 = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]

        noPEC = False

            #datafitting here
        try:
            PEC = inp(data['en'], pecDict[j])
            print(j)
        except KeyError:
            PEC = scipy.ones(data['en'].shape)
            noPEC = True

        for i in temp:
            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,j]*PEC)
            output2[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,j])

        for i in temp:
            pid = p == i

            lims = scipy.logical_and(data['en'][p[pid]] > 1e3, data['en'][p[pid]] < 2e4)
            idx = scipy.logical_and(output[p[pid]] != 0, expdata[pid] !=0)
            idx2 = scipy.logical_and(output2[p[pid]] != 0, expdata[pid] !=0)

            idx12 = scipy.logical_and(lims, idx)
            idx22 = scipy.logical_and(lims, idx2)

            differ = scipy.log(output[p[pid]][idx12]/output.max())-scipy.log(expdata[pid][idx12])
            differ2 = scipy.log(output2[p[pid]][idx22]/output2.max())-scipy.log(expdata[pid][idx22])

            variances[i] = scipy.var(differ)

    plt.loglog(data['en'],variances,'.',color='#0E525A',markersize=10)

    plt.xlabel(r'$T_e$ [eV]')
    plt.ylabel(r'Var$(\log(I_{CSXR}/c_W \sum M))$')
    #leg = plt.legend(numpoints=1,fontsize=18,title=r'placeholder')
    #plt.setp(leg.get_title(),fontsize=14)
    plt.subplots_adjust(bottom=.12)
    plt.show()



        #plt.loglog(data['en'],output2/output2.max(),lw=3,color='magenta',alpha=alpha,linestyle='--',label=r'W$^{45+}$')
        #plt.loglog(data['en'],output/output.max(),lw=3,color='magenta',alpha=alpha,linestyle='-',label=r'_nolegend')
        #plt.xlabel(r'$T_e$ [eV]')
        #plt.ylabel(r'const$\cdot I_{CSXR} / c_W \sum M$')

    return output2[p], expdata

def scatterplotPEC(M, name, nozero=False, k=2.5e29, useall=False, lim=1e-2, conc='c_w_l',loc='data_to_Ian.txt'):
    """ log plot of the data with only PEC data"""
    if useall:
        val, c_w_l, idx = conditionVal2(name=name, nozero=nozero, conc=conc)
    else:
        val, c_w_l, idx = conditionVal(name=name, nozero=nozero, conc=conc)
              
    data = scipy.io.readsav('W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave')
    y = time.time()

    maxT = HighTe(M)
    #plt.loglog(maxT,val/c_w_l/scipy.sum(M,axis=1)*1e30,'.',alpha=.01)
  
    expdata = val/c_w_l/scipy.sum(M,axis=1)*k

    plt.scatter(maxT,expdata,c=c_w_l,cmap='viridis', rasterized=True, norm=LogNorm(),alpha=.25)
    plt.gca().set_yscale('log')
    plt.gca().set_xscale('log')
    plt.gca().set_ylim(1e-4,1e1)
    plt.gca().set_xlim(1e3,2e4)
   
    cmap = plt.cm.get_cmap('viridis')
    cmap.set_under('white')
 
    inp = adsre.lineInfo(loc)
    #plt.plot([],[],color='k',lw=2,label=r'with PEC') 


    if True:
        alpha=.8
        idx = scipy.logical_and(data['en'] > 1e3, data['en'] < 5e4)
        output = scipy.zeros(data['en'].shape)
        output2 = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        for i in temp:
            PEC = inp(data['en'], 0)
            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,46]*PEC)
            output2[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,46])

        plt.loglog(data['en'],output/output.max(),lw=3,color='crimson',alpha=alpha,linestyle='--',label=r'W$^{46+}$')

        #plt.loglog(data['en'],output/output.max(),lw=3,color='crimson',alpha=alpha,linestyle='-',label=r'_nolegend_')   
        output = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        for i in temp:
            PEC = inp(data['en'], 3)
            
            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,45]*PEC)
            output2[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,45])


        plt.loglog(data['en'],output/output.max(),lw=3,color='magenta',alpha=alpha,linestyle='--',label=r'W$^{45+}$')
        #plt.loglog(data['en'],output/output.max(),lw=3,color='magenta',alpha=alpha,linestyle='-',label=r'_nolegend')

        output = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        for i in temp:
            PEC = inp(data['en'], 6)

            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,44]*PEC)
            output2[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,44])

        plt.loglog(data['en'],output/output.max(),lw=3,color='cyan',alpha=alpha,linestyle='--',label=r'W$^{44+}$')
        #plt.loglog(data['en'],output/output.max(),lw=3,color='cyan',alpha=alpha,linestyle='-',label=r'_nolegend_')

        plt.xlabel(r'$T_e$ [eV]')
        plt.ylabel(r'const$\cdot I_{CSXR} / c_W \sum M$')

    colorbar = plt.colorbar()
    colorbar.ax.set_ylabel('$c_W$')
    leg = plt.legend(loc=4,fontsize=20,title=r'\underline{\hspace{1em} \emph{with PEC} \hspace{1em}}')
    plt.setp(leg.get_title(),fontsize=14)
    plt.subplots_adjust(bottom=.12,right=1.)

    #plt.show()

def visualizeCounts(M, name, nozero=False, k=2.5e29, useall=False, lim=1e-2, conc='c_w_l',loc='data_to_Ian.txt'):
    """ sort the data like the PEC graphs, and then calculate parameters of interest"""
    if useall:
        val, c_w_l, idx = conditionVal2(name=name, nozero=nozero, conc=conc)
    else:
        val, c_w_l, idx = conditionVal(name=name, nozero=nozero, conc=conc)
              
    data = scipy.io.readsav('W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave')
    y = time.time()

    maxT = HighTe(M)
    Tsum = scipy.sum(M,axis=1)
    #plt.loglog(maxT,val/c_w_l/scipy.sum(M,axis=1)*1e30,'.',alpha=.01)
    xax = scipy.logspace(2,5,201)
    yax = scipy.logspace(-5,1,101)
    histdata = scipy.zeros((len(xax),len(yax)))
    xtemp = xax[1:]/2.+xax[:-1]/2.
    ytemp = yax[1:]/2.+yax[:-1]/2.

    expdata = val/c_w_l/scipy.sum(M,axis=1)*k

    idx1 = scipy.searchsorted(xax,maxT)
    idx2 = scipy.searchsorted(yax,expdata)

    #this is horrible, but its the easiest and quickest solution

    for k in xrange(len(xax)):
        for l in xrange(len(yax)):
            histtemp = scipy.mean(Tsum[scipy.logical_and(idx1 == k, idx2 == l)])
            if not scipy.isnan(histtemp):
                histdata[k,l] = abs(histtemp)


    #histdata, xed, yed = scipy.histogram2d(maxT,val/c_w_l/scipy.sum(M,axis=1)*k,bins=[xax,yax])
    plt.pcolormesh(xax, yax, histdata.T, cmap='viridis', rasterized=True, norm=LogNorm(),vmin=1e38)
    plt.gca().set_yscale('log')
    plt.gca().set_xscale('log')
    plt.gca().set_ylim(1e-4,1e1)
    plt.gca().set_xlim(1e3,2e4)
   
    cmap = plt.cm.get_cmap('viridis')
    cmap.set_under('white')
 
    inp = adsre.lineInfo(loc)
    #plt.plot([],[],color='k',lw=2,label=r'with PEC') 


    if True:
        alpha=.8
        idx = scipy.logical_and(data['en'] > 1e3, data['en'] < 5e4)
        output = scipy.zeros(data['en'].shape)
        output2 = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        for i in temp:
            PEC = inp(data['en'], 0)
            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,46]*PEC)
            output2[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,46])

        plt.loglog(data['en'],output/output.max(),lw=3,color='crimson',alpha=alpha,linestyle='--',label=r'W$^{46+}$')

        #plt.loglog(data['en'],output/output.max(),lw=3,color='crimson',alpha=alpha,linestyle='-',label=r'_nolegend_')   
        output = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        for i in temp:
            PEC = inp(data['en'], 3)
            
            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,45]*PEC)
            output2[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,45])


        plt.loglog(data['en'],output/output.max(),lw=3,color='magenta',alpha=alpha,linestyle='--',label=r'W$^{45+}$')
        #plt.loglog(data['en'],output/output.max(),lw=3,color='magenta',alpha=alpha,linestyle='-',label=r'_nolegend')

        output = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        for i in temp:
            PEC = inp(data['en'], 6)

            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,44]*PEC)
            output2[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,44])

        plt.loglog(data['en'],output/output.max(),lw=3,color='cyan',alpha=alpha,linestyle='--',label=r'W$^{44+}$')
        #plt.loglog(data['en'],output/output.max(),lw=3,color='cyan',alpha=alpha,linestyle='-',label=r'_nolegend_')

        plt.xlabel(r'$T_e$ [eV]')
        plt.ylabel(r'const$\cdot I_{CSXR} / c_W \sum M$')

    colorbar = plt.colorbar()
    colorbar.ax.set_ylabel('$\sum M$ [m$^{-5}$]')
    leg = plt.legend(loc=4,fontsize=20,title=r'\underline{\hspace{1em} \emph{with PEC} \hspace{1em}}')
    plt.setp(leg.get_title(),fontsize=14)
    plt.subplots_adjust(bottom=.12,right=.99)

    return histdata
    
    #plt.show()

def visualizeCoVar(M, name, covar, nozero=False, k=2.5e29, lim=1e-2, conc='c_w_l',loc='data_to_Ian.txt'):
    """ sort the data like the PEC graphs, and then calculate parameters of interest"""
 
    val, c_w_l, idx, good = conditionVal4(name=name, nozero=nozero, conc=conc)
    val4 = loadData(covar)
              
    data = scipy.io.readsav('W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave')
    y = time.time()

    maxT = HighTe(M)

    #plt.loglog(maxT,val/c_w_l/scipy.sum(M,axis=1)*1e30,'.',alpha=.01)
    xax = scipy.logspace(2,5,201)
    yax = scipy.logspace(-5,1,101)
    histdata = scipy.zeros((len(xax),len(yax)))
    xtemp = xax[1:]/2.+xax[:-1]/2.
    ytemp = yax[1:]/2.+yax[:-1]/2.

    expdata = val/c_w_l/scipy.sum(M,axis=1)*k

    idx1 = scipy.searchsorted(xax,maxT)
    idx2 = scipy.searchsorted(yax,expdata)

    #this is horrible, but its the easiest and quickest solution

    for k in xrange(len(xax)):
        for l in xrange(len(yax)):
            histtemp = scipy.mean(val4[scipy.logical_and(idx1 == k, idx2 == l)])
            if not scipy.isnan(histtemp):
                histdata[k,l] = abs(histtemp)


    #histdata, xed, yed = scipy.histogram2d(maxT,val/c_w_l/scipy.sum(M,axis=1)*k,bins=[xax,yax])
    plt.pcolormesh(xax, yax, histdata.T, cmap='viridis', rasterized=True, norm=LogNorm())
    plt.gca().set_yscale('log')
    plt.gca().set_xscale('log')
    plt.gca().set_ylim(1e-4,1e1)
    plt.gca().set_xlim(1e3,2e4)
   
    cmap = plt.cm.get_cmap('viridis')
    cmap.set_under('white')
 
    inp = adsre.lineInfo(loc)
    #plt.plot([],[],color='k',lw=2,label=r'with PEC') 


    if True:
        alpha=.8
        idx = scipy.logical_and(data['en'] > 1e3, data['en'] < 5e4)
        output = scipy.zeros(data['en'].shape)
        output2 = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        for i in temp:
            PEC = inp(data['en'], 0)
            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,46]*PEC)
            output2[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,46])

        plt.loglog(data['en'],output/output.max(),lw=3,color='crimson',alpha=alpha,linestyle='--',label=r'W$^{46+}$')

        #plt.loglog(data['en'],output/output.max(),lw=3,color='crimson',alpha=alpha,linestyle='-',label=r'_nolegend_')   
        output = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        for i in temp:
            PEC = inp(data['en'], 3)
            
            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,45]*PEC)
            output2[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,45])


        plt.loglog(data['en'],output/output.max(),lw=3,color='magenta',alpha=alpha,linestyle='--',label=r'W$^{45+}$')
        #plt.loglog(data['en'],output/output.max(),lw=3,color='magenta',alpha=alpha,linestyle='-',label=r'_nolegend')

        output = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        for i in temp:
            PEC = inp(data['en'], 6)

            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,44]*PEC)
            output2[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,44])

        plt.loglog(data['en'],output/output.max(),lw=3,color='cyan',alpha=alpha,linestyle='--',label=r'W$^{44+}$')
        #plt.loglog(data['en'],output/output.max(),lw=3,color='cyan',alpha=alpha,linestyle='-',label=r'_nolegend_')

        plt.xlabel(r'$T_e$ [eV]')
        plt.ylabel(r'const$\cdot I_{CSXR} / c_W \sum M$')

    colorbar = plt.colorbar()
    colorbar.ax.set_ylabel('line counts')
    leg = plt.legend(loc=4,fontsize=20,title=r'\underline{\hspace{1em} \emph{with PEC} \hspace{1em}}')
    plt.setp(leg.get_title(),fontsize=14)
    plt.subplots_adjust(bottom=.12,right=.99)

    return histdata
    
    #plt.show()

def visualize2PEC2star(M, name, nozero=False, k=2.5e29, lim=1e-2, conc='c_w_l',loc='data_to_Ian.txt'):
    """ log plot of the data with only PEC data"""

    val, c_w_l, idx, good = conditionVal4(name=name, nozero=nozero, conc=conc)
    width = loadData('width7')[good]

    good2 = scipy.logical_and(width > 2e-5, width < 3e-4)


    data = scipy.io.readsav('W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave')
    y = time.time()

    maxT = HighTe(M)
    #plt.loglog(maxT,val/c_w_l/scipy.sum(M,axis=1)*1e30,'.',alpha=.01)
    xax = scipy.logspace(2,5,201)
    yax = scipy.logspace(-5,1,101)
    xtemp = xax[1:]/2.+xax[:-1]/2.
    ytemp = yax[1:]/2.+yax[:-1]/2.

    Y,X = scipy.meshgrid(xtemp, ytemp)
    histdata, xed, yed = scipy.histogram2d(maxT[good2],width[good2]*val[good2]/c_w_l[good2]/scipy.sum(M[good2],axis=1)*k,bins=[xax,yax])
    extent = [xed[0], xed[-1], yed[0], yed[-1]]
    plt.pcolormesh(xax,yax,histdata.T, cmap='viridis', rasterized=True, norm=LogNorm())
    plt.gca().set_yscale('log')
    plt.gca().set_xscale('log')
    plt.gca().set_ylim(1e-4,1e1)
    plt.gca().set_xlim(1e3,2e4)
   
    cmap = plt.cm.get_cmap('viridis')
    cmap.set_under('white')
 
    inp = adsre.lineInfo(loc)
    #plt.plot([],[],color='k',lw=2,label=r'with PEC') 


    if True:
        alpha=.8
        idx = scipy.logical_and(data['en'] > 1e3, data['en'] < 5e4)
        output = scipy.zeros(data['en'].shape)
        output2 = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        for i in temp:
            PEC = inp(data['en'], 0)
            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,46]*PEC)
            output2[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,46])

        plt.loglog(data['en'],output/output.max(),lw=3,color='crimson',alpha=alpha,linestyle='--',label=r'W$^{46+}$')

        #plt.loglog(data['en'],output/output.max(),lw=3,color='crimson',alpha=alpha,linestyle='-',label=r'_nolegend_')   
        output = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        for i in temp:
            PEC = inp(data['en'], 3)
            
            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,45]*PEC)
            output2[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,45])


        plt.loglog(data['en'],output/output.max(),lw=3,color='magenta',alpha=alpha,linestyle='--',label=r'W$^{45+}$')
        #plt.loglog(data['en'],output/output.max(),lw=3,color='magenta',alpha=alpha,linestyle='-',label=r'_nolegend')

        output = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        for i in temp:
            PEC = inp(data['en'], 6)

            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,44]*PEC)
            output2[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,44])

        plt.loglog(data['en'],output/output.max(),lw=3,color='cyan',alpha=alpha,linestyle='--',label=r'W$^{44+}$')
        #plt.loglog(data['en'],output/output.max(),lw=3,color='cyan',alpha=alpha,linestyle='-',label=r'_nolegend_')

        plt.xlabel(r'$T_e$ [eV]')
        plt.ylabel(r'const$\cdot I_{CSXR} / c_W \sum M$')

    colorbar = plt.colorbar()
    colorbar.ax.set_ylabel('datapoints in bin')
    leg = plt.legend(loc=4,fontsize=20,title=r'\underline{\hspace{1em} \emph{with PEC} \hspace{1em}}')
    plt.setp(leg.get_title(),fontsize=14)
    plt.subplots_adjust(bottom=.12,right=1.)

    #plt.show()

def visualizeWidth(M, name, nozero=False, k=2e29, useall=False, lim=1e-2, conc='c_w_l'):

    val, c_w_l, idx, good = conditionVal4(name=name, nozero=nozero, conc=conc)
    val = loadData('width7')[good]

    good2 = scipy.logical_and(val > 2e-5, val < 3e-4)

    data = scipy.io.readsav('W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave')
    y = time.time()

    maxT = HighTe(M)
    #plt.loglog(maxT,val/c_w_l/scipy.sum(M,axis=1)*1e30,'.',alpha=.01)
    xax = scipy.logspace(2,5,201)
    yax = scipy.logspace(-5,-3,101)
    xtemp = xax[1:]/2.+xax[:-1]/2.
    ytemp = yax[1:]/2.+yax[:-1]/2.

    Y,X = scipy.meshgrid(xtemp, ytemp)
    histdata, xed, yed = scipy.histogram2d(maxT[good2],val[good2],bins=[xax,yax])
    plt.pcolormesh(xax,yax,histdata.T,norm=LogNorm(), cmap='viridis', rasterized=True)
    plt.gca().set_yscale('log')
    plt.gca().set_xscale('log')
    plt.gca().set_ylim(1e-5,1e-3)
    plt.gca().set_xlim(1e3,2e4)
    


    plt.xlabel(r'$T_e$ [eV]')
    plt.ylabel(r'width')

    colorbar = plt.colorbar()
    colorbar.ax.set_ylabel('datapoints in bin')
    plt.subplots_adjust(bottom=.12,right=1.)

    #plt.show()
