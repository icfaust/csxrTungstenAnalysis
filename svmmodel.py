import sklearn
import sklearn.svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import scipy
import scipy.io
import scipy.interpolate
import matplotlib.pyplot as plt
import adsre
import time
import testM
import sqlite3
import pickle
import numpy.random
from matplotlib.colors import LogNorm
from matplotlib import rc

#This program extracts at SVM reduced fit from the forward model assumption of T_emax 
#dependence, it then uses the forward model to yield a representation of f_Z
#The values are then cross-validated using the 2014-2016 dataset, to yield the best fit


semiEmperical = scipy.array([.3909,.39316,.39331,.39374,.39492,.3955,.3963,.39648,.3966,.39695,.3975,.39887,.39943])
rc('text',usetex=True)
rc('text.latex',preamble='\usepackage{xcolor}')
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
#rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern Sans serif']})
rc('font',size=20)
disp = 86363.6363


############################################################
#                  data loading scripts                    #
############################################################


def loadData(name, database='SIFdataMLE.db'):
    """ loads data from the fitted database of CSXR data"""
    a = sqlite3.connect(database)
    b = a.cursor()
    return scipy.squeeze(b.execute('SELECT '+name+' FROM shots2016').fetchall())

def loadM(idx, database='SIFreduced.db'):
    a = sqlite3.connect(database)
    b = a.cursor()
    temp = scipy.squeeze(b.execute('SELECT * FROM shots2015').fetchall())
    order = scipy.searchsorted(temp[:,0],idx)
    return temp[order,1:]

def conditionVal(name='val7', nozero=False, conc='c_w_l'):
    #this is a program which extracts the useful data from the CSXR data

    val = loadData(name)
    idx = loadData('id')
    c_w_l = loadData(conc)
    shot = loadData('shot')

    base = loadData('baseline')
    width = loadData('width'+name[-1])

    fiducial = (base+val/scipy.sqrt(scipy.pi)/width)*123+1000

  
    #c_w_l must have a value below 1e-2 and nonzero
    good = c_w_l > 0 #this had to be done based on how the M matrix was generated (for all non-zero c_w_l values)

    idx = idx[good]
    val = val[good]
    shot = shot[good]
    c_w_l = c_w_l[good]


    #c_w_l must have a value below 1e-2 and nonzero
    good = scipy.logical_and(scipy.logical_and(fiducial[good] < 6e4, c_w_l < 1e-2), val*disp > 1e0)

    idx = idx[good]
    val = val[good]
    shot = shot[good]
    c_w_l = c_w_l[good]


    #val[val < 1.] = 0.

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

def sample(val,samples=None):
    temp = scipy.arange(len(val))

    if samples is None:
        samples = temp.shape
            
    idx = numpy.random.choice(temp, samples)
    return idx

def patch(ydata, xdata=[2.2e3,9e3], name='W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave'):
    if len(xdata) == 2:
        te = scipy.io.readsav(name)['en']
        xdata = te[scipy.logical_and(te > xdata[0], te < xdata[1])]

    #print(len(xdata))
        
    output = scipy.interpolate.interp1d(scipy.log(xdata),
                                        scipy.log(ydata),
                                        bounds_error=False)
    return output

############################################################
#                  Fitting data routines                   #
############################################################

# This is where the SVM work is generated

def SVMtest(maxT, sumM, val, c_w_l, C=1e3,gamma=1e-1, k=2.2e35, lims=[2.2e3,9e3], reduced=None):


    #val, c_w_l, idx = conditionVal(name=name, nozero=nozero, conc=conc)
    temp = val/c_w_l/sumM*k
    print(maxT.shape, temp.shape)
    #data = scipy.io.readsav('W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave')
    #te = data['en']
    #idx2 = scipy.logical_and(te > lims[0], te < lims[1])

    y = time.time()

    if reduced is None:
        xData = scipy.log(scipy.atleast_2d(maxT).T)
        yData = scipy.log(temp)
    else:
        xData = scipy.log(scipy.atleast_2d(maxT[reduced]).T)
        yData = scipy.log(temp[reduced])

    svm = sklearn.svm.SVR(cache_size=7000)#,C=C,gamma=gamma)
    pipe = Pipeline([('scale',StandardScaler()),('svm',svm)])
    pipe.set_params(svm__C=C,svm__gamma=gamma)

    pipe.fit(xData,yData)
    print(time.time()-y)

    return pipe
    #return scipy.exp(pipe.predict(scipy.log(te[idx2][None].T)))


def gridSearch(maxT, sumM, name,loc='svmcrossvalid.p',reduction=50,k=2.2e35,gamma=[-3,3], C=[0,6], nozero=False, conc='c_w_l', lims=[2.2e3,9e3]):

    val, c_w_l, idx = conditionVal(name=name, nozero=nozero, conc=conc)
    idx1 = sample(val,len(val)/reduction)
    idx2 = sample(val,len(val)/reduction) #very small subsamples started for testing algos
    index0 = scipy.logspace(gamma[0],gamma[1],int(abs(gamma[0]-gamma[1])+1))
    index1 = scipy.logspace(C[0],C[1],int(abs(C[0]-C[1])+1))

    data = scipy.io.readsav('W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave')
    te = data['en']
    idx2 = scipy.logical_and(te > lims[0], te < lims[1])


    temp = val[idx2]/c_w_l[idx2]/sumM[idx2]*k


    output = scipy.zeros((len(index0),len(index1)))
    output2 = scipy.zeros((len(index0),len(index1),len(te[idx2])))



    for i in xrange(len(index0)):
        for j in xrange(len(index1)):
            print(i,j)
            pipe = SVMtest(maxT, sumM, val, c_w_l, reduced=idx1, gamma=index0[i], C=index1[j],k=k)
            output[i,j] = pipe.score(scipy.log(scipy.atleast_2d(maxT[idx2]).T),scipy.log(temp))
            output2[i,j] = scipy.exp(pipe.predict(scipy.log(scipy.atleast_2d(te[idx2]).T)))

    pickle.dump([output,output2],open(loc,'wb'))
    return output,output2

def bootstrap(maxT, sumM, name,loc='svmbootstrap.p',reduction=50,k=2.2e35,gamma=1e-1, C=1e5, nozero=False, conc='c_w_l', lims=[2.2e3,9e3],pts=1001):

    val, c_w_l, idx = conditionVal(name=name, nozero=nozero, conc=conc)

    idx1 = sample(val,len(val)/reduction)
    data = scipy.io.readsav('W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave')
    te = data['en']
    idx2 = scipy.logical_and(te > lims[0], te < lims[1])
    #idxtest = maxT > lims[0]

    output = scipy.zeros((pts,len(te[idx2])))


    temp = val/c_w_l/sumM*k


    for i in xrange(pts):
        print(i)
        idx1 = sample(val,len(val)/reduction)
        pipe = SVMtest(maxT, sumM, val, c_w_l, reduced=idx1, gamma=gamma, C=C, k=k)
        output[i] = scipy.exp(pipe.predict(scipy.log(scipy.atleast_2d(te[idx2]).T)))

    pickle.dump(output,open(loc,'wb'))
    return output

def genHatM(lims=[2.3e3,9e3]):
    #uses the Te
    data = scipy.io.readsav('W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave')

    te = data['en']
    idx2 = scipy.logical_and(te > lims[0], te < lims[1])
    output = scipy.zeros((len(te[idx2]),len(te[idx2])))
    for i in xrange(len(te[idx2])):
        temp = testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][idx2][i]), .125)
        output[i] = temp[idx2]

    return output


############################################################
#                    Plotting routines                     #
############################################################              

def visualize2PEC(maxT, sumM, name, nozero=False, k=2.2e35, lim=1e-2, conc='c_w_l',loc='data_to_Ian.txt'):
    """ log plot of the data comparison with and without PEC data"""
    val, c_w_l, idx = conditionVal(name=name, nozero=nozero, conc=conc)
              
    data = scipy.io.readsav('W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave')
    y = time.time()

    xax = scipy.logspace(2,5,201)
    yax = scipy.logspace(-5,1,101)
    xtemp = xax[1:]/2.+xax[:-1]/2.
    ytemp = yax[1:]/2.+yax[:-1]/2.

    Y,X = scipy.meshgrid(xtemp, ytemp)
    histdata, xed, yed = scipy.histogram2d(maxT,val/c_w_l/sumM*k,bins=[xax,yax])
    extent = [xed[0], xed[-1], yed[0], yed[-1]]
    plt.pcolormesh(xax,yax,histdata.T, cmap='viridis', rasterized=True, norm=LogNorm())
    plt.gca().set_yscale('log')
    plt.gca().set_xscale('log')
    plt.gca().set_ylim(1e-4,1e1)
    plt.gca().set_xlim(1e3,2e4)
   
    cmap = plt.cm.get_cmap('viridis')
    cmap.set_under('white')
 
    inp = adsre.lineInfo(loc)

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
    colorbar.ax.set_ylabel('datapoint density (a.u.)')
    leg = plt.legend(loc=4,fontsize=20,title=r'\underline{\emph{solid} - \emph{with PEC}}')
    plt.setp(leg.get_title(),fontsize=14)
    plt.subplots_adjust(bottom=.12,right=1.)

    #plt.show()

def visualize2PEC2(maxT, sumM, name, nozero=False, k=2.2e35, useall=False, lim=1e-2, conc='c_w_l',loc='data_to_Ian.txt',idx2=None):
    """ log plot of the data with only PEC data"""
    val, c_w_l, idx = conditionVal(name=name, nozero=nozero, conc=conc)
              
    data = scipy.io.readsav('W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave')
    y = time.time()

    xax = scipy.logspace(2,5,201)
    yax = scipy.logspace(-5,1,101)
    xtemp = xax[1:]/2.+xax[:-1]/2.
    ytemp = yax[1:]/2.+yax[:-1]/2.

    Y,X = scipy.meshgrid(xtemp, ytemp)
    if idx2 is None:
        histdata, xed, yed = scipy.histogram2d(maxT,val/c_w_l/sumM*k,bins=[xax,yax])
    else:
        histdata, xed, yed = scipy.histogram2d(maxT[idx2],(val/c_w_l/sumM*k)[idx2],bins=[xax,yax])
        
    extent = [xed[0], xed[-1], yed[0], yed[-1]]
    plt.pcolormesh(xax,yax,histdata.T, cmap='viridis', rasterized=True,vmin=1.,norm=LogNorm())
    plt.gca().set_yscale('log')
    plt.gca().set_xscale('log')
    plt.gca().set_ylim(1e-4,1e1)
    plt.gca().set_xlim(1e3,2e4)
   
    cmap = plt.cm.get_cmap('viridis')
    cmap.set_under('white')
 
    inp = adsre.lineInfo(loc)

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

        output = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        for i in temp:
            PEC = inp(data['en'], 3)
            
            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,45]*PEC)
            output2[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,45])


        plt.loglog(data['en'],output/output.max(),lw=3,color='magenta',alpha=alpha,linestyle='--',label=r'W$^{45+}$')

        output = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        for i in temp:
            PEC = inp(data['en'], 6)

            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,44]*PEC)
            output2[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,44])

        plt.loglog(data['en'],output/output.max(),lw=3,color='cyan',alpha=alpha,linestyle='--',label=r'W$^{44+}$')

        plt.xlabel(r'$T_e$ [eV]')
        plt.ylabel(r'const$\cdot I_{CSXR} / c_W \sum M$')

    colorbar = plt.colorbar()
    colorbar.ax.set_ylabel('datapoint density (a.u.)')
    leg = plt.legend(loc=4,fontsize=20,title=r'\underline{\hspace{1em} \emph{with PEC} \hspace{1em}}')
    plt.setp(leg.get_title(),fontsize=14)
    plt.subplots_adjust(bottom=.12,right=1.)


def visualize2SVM(maxT, sumM, name, nozero=False, k=2.2e35, useall=False, lims=[2.2e3,9e3], minval=[2,5], conc='c_w_l',loc='svmcrossvalid.p',idx2=None):
    """ log plot of the data with only PEC data"""
    val, c_w_l, idx = conditionVal(name=name, nozero=nozero, conc=conc)
              
    data = scipy.io.readsav('W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave')
    y = time.time()

    xax = scipy.logspace(2,5,201)
    yax = scipy.logspace(-5,1,101)
    xtemp = xax[1:]/2.+xax[:-1]/2.
    ytemp = yax[1:]/2.+yax[:-1]/2.

    Y,X = scipy.meshgrid(xtemp, ytemp)
    if idx2 is None:
        histdata, xed, yed = scipy.histogram2d(maxT,val/c_w_l/sumM*k,bins=[xax,yax])
    else:
        histdata, xed, yed = scipy.histogram2d(maxT[idx2],(val/c_w_l/sumM*k)[idx2],bins=[xax,yax])
        
    extent = [xed[0], xed[-1], yed[0], yed[-1]]
    plt.pcolormesh(xax,yax,histdata.T, cmap='viridis', rasterized=True,vmin=1.,norm=LogNorm())
    plt.gca().set_yscale('log')
    plt.gca().set_xscale('log')
    plt.gca().set_ylim(1e-4,1e1)
    plt.gca().set_xlim(1e3,2e4)
   
    cmap = plt.cm.get_cmap('viridis')
    cmap.set_under('white')
 
    idx3 = scipy.logical_and(data['en'] > lims[0], data['en'] < lims[1])
    data2 = pickle.load(open(loc,'rb')) 
    fz = data2[1][minval[0]][minval[1]]


    print(len(fz))
    print(data['en'][idx3].shape)

    plt.loglog(data['en'][idx3],fz/fz.max(),lw=3.5,color='darkorange',linestyle='-',label='SVM fit')

    plt.xlabel(r'$T_e$ [eV]')
    plt.ylabel(r'const$\cdot I_{CSXR} / c_W \sum M$')

    colorbar = plt.colorbar()
    colorbar.ax.set_ylabel('datapoint density (a.u.)')
    leg = plt.legend(loc=4,fontsize=20,title=r'\underline{\hspace{1em} C=10$^4$, $\gamma=.1$ \hspace{1em}}')
    plt.setp(leg.get_title(),fontsize=14)
    plt.subplots_adjust(bottom=.12,right=1.)

def plotHatM(lims=[2.2e3,9e3]):

    dataset = scipy.io.readsav('W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave')    
    Te = dataset['en']

    idx = scipy.logical_and(Te > lims[0], Te < lims[1])

    grid = genHatM(lims)

    plt.pcolormesh(Te[idx],Te[idx],grid*1e3,cmap='viridis')
    
    plt.xticks([3e3,4e3,5e3,6e3,7e3,8e3],['3','4','5','6','7','8'])
    plt.yticks([3e3,4e3,5e3,6e3,7e3,8e3],['3','4','5','6','7','8'])  

    plt.axis('tight')

    plt.xlabel('$T_e$ keV')
    plt.ylabel('$T_e$ keV')

    colorbar = plt.colorbar()
    colorbar.ax.set_ylabel('$\hat{M}$ weighting matrix ($10^{-3}$)')
    
    
    plt.show()

                   
def regSolutionExample(minval=[2,5],loc='svmcrossvalid.p',k=3e35, lims=[2.2e3,9e3], lims2=[2.3e3,7e3], alpha=.8,err=None):

    dataset = scipy.io.readsav('W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave')    
    Te = dataset['en']

    idx = scipy.logical_and(Te > lims[0], Te < lims[1])
    idxin = idx
    idx2 = scipy.logical_and(Te[idx] > lims2[0], Te[idx] < lims2[1])
    idx = scipy.logical_and(Te > lims2[0], Te < lims2[1])
    data = pickle.load(open(loc,'rb')) 

    fz = data[1][minval[0]][minval[1]]
    Mhat = scipy.linalg.inv(genHatM(lims))
    fz = scipy.dot(Mhat,fz)

   
    #didx = scipy.logical_and(Te > 1e3, Te < 5e4)
    loc2='data_to_Ian.txt'
    inp = adsre.lineInfo(loc2)
    PEC = inp(dataset['en'],6)
    data1 = dataset['abundance'][:,44]*PEC
    data1 = data1/data1.max()

    PEC = inp(dataset['en'],3)
    data2 = dataset['abundance'][:,45]*PEC
    data2 = data2/data2.max()

    PEC = inp(dataset['en'],0)
    data3 = dataset['abundance'][:,46]*PEC
    data3 = data3/data3.max()

    #if not err is None:
    #    lb = data[minval][1].x[idx2]-err[idx2]
    #    lb = lb/2 + abs(lb)/2+1e-50
    #    ub = data[minval][1].x[idx2]+err[idx2]
#
#        plt.fill_between(Te[idx],
#                         lb/data[minval][1].x[idx2].max(),
#                         ub/data[minval][1].x[idx2].max(),
#                         facecolor='#0E525A',
#                         alpha=.4,
#                         color="none")
#    else:
#        plt.fill_between(Te[idx],
#                         sigma[idx2]*(data[minval][1].x[idx2])/data[minval][1].x[idx2].max(),
#                         1./sigma[idx2]*(data[minval][1].x[idx2])/data[minval][1].x[idx2].max(),
#                         facecolor='#0E525A',
#                         alpha=.4,
#                         color="none")
#
    #plt.loglog(Te[idx],1./sigma[idx2]*(data[minval][1].x[idx2])/data[minval][1].x[idx2].max(),color='gold',lw=3,label='_nolabel_')    
    #plt.loglog(Te[idx],sigma[idx2]*(data[minval][1].x[idx2])/data[minval][1].x[idx2].max(),color='gold',lw=3,label='_nolabel_')


    plt.loglog(Te[idxin][idx2],fz[idx2]/fz[idx2].max(),lw=3,color='darkorange',label='_nolabel_')

    plt.loglog(Te,data3,lw=3,color='crimson',alpha=alpha,linestyle='--',label=r'W$^{46+}$')
    plt.loglog(Te,data2,lw=3,color='magenta',alpha=alpha,linestyle='--',label=r'W$^{45+}$')
    plt.loglog(Te,data1,lw=3,color='cyan',alpha=alpha,linestyle='--',label=r'W$^{44+}$')

    plt.xlim(2e3,9e3)
    plt.ylim(1e-2,3e0)
    plt.xlabel('$T_e$ [keV]')
    plt.ylabel('PEC$\cdot f_Z/$max(PEC$\cdot f_Z$)')
    plt.xticks([2e3,3e3,4e3,5e3,6e3,7e3,8e3,9e3],['2','3','4','5','6','7','8','9'])
    leg = plt.legend(loc=8,fontsize=20,title=r'\underline{\hspace{.5em} \emph{solid} - \emph{fit of PEC$\cdot f_Z$}\hspace{.5em} }')   
    plt.setp(leg.get_title(),fontsize=14)
    plt.subplots_adjust(bottom=.12)
    plt.show()
