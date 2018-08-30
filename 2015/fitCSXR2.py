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
import pickle

semiEmperical = scipy.array([.3909,.39316,.39331,.39374,.39492,.3955,.3963,.39648,.3966,.39695,.3975,.39887,.39943])

rc('text',usetex=True)
rc('text.latex',preamble='\usepackage{xcolor}')
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
#rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern Sans serif']})
rc('font',size=20)
disp = 86363.6363

############################################################
#                      temporary funcs                     #
############################################################

def gtest(fz, val, c_W, M):
    delta = 1e-6
    output = scipy.zeros(fz.shape)
    start = objectiveLogJac(fz, val, c_W, M)
    for i in xrange(len(fz)):
        print(i),
        fz[i] += delta
        p1 = fullObjectiveLog(fz, val, c_W, M)
        fz[i] -= 2*delta
        p0 = fullObjectiveLog(fz, val, c_W, M)
        fz[i] += delta
        output[i] = (p1-p0)/(2*delta)

    return output, start


############################################################
#                 optimization functions                   #
############################################################

def gaussian(x,inp):
    return inp[0]*scipy.exp(-1*pow((x-inp[1])/inp[2],2))

def lognorm(x,inp):
    return (inp[0]/x)*scipy.exp(-1*pow((scipy.log(x)-inp[1])/inp[2],2))

def dotPositive(mat, x):
    """ this returns values for positive values of the matrix multiplication, necessary for negativity 2nd derivative constraint""" 
    temp = scipy.dot(mat,x)
    output = (temp + abs(temp))/2
    return output

def genI(inp, Te, c_W, M):
    """ generate the synthetic expected CSXR emission (not including the fractional abundance)"""
    return c_W*scipy.dot(M, lognorm(Te, inp))

def objectiveLog(inp, val, Te, c_W, M):
    """ objective function for minimization with logarithm of the data, assumes log-normal distribution"""
    return scipy.sqrt(scipy.sum(pow(scipy.log(val) - scipy.log(genI(inp, Te, c_W, M)),2)))/2.

def objective(inp, val, Te, c_W, M):
    """ objective function for minimization, in this case is the RMSE and assumes normal errors"""
    return scipy.sqrt(scipy.sum(pow(val - genI(inp, Te, c_W, M),2)))/2.

def fullObjective(fz, val, c_W, M):
    """ instead of a parabolic assumption used in objective and objectiveLog, this uses the full fz definition"""
    return scipy.sum(pow(val - c_W*scipy.dot(M,fz),2))/2.

def fullObjectiveLog(fz, val, c_W, M):
    """ instead of a parabolic assumption used in objective and objectiveLog, this uses the full fz definition with log"""
    temp = c_W*scipy.dot(M, fz)
    idx = temp != 0
    return scipy.sum(pow(scipy.log(val[idx]) - scipy.log(temp[idx]), 2))/2.

def objectiveJac(fz, val, c_W, M):
    output = scipy.zeros(fz.shape)
    valstar = c_W*scipy.dot(M, fz)
    # I am going to do this in a for loop, because M is huge and I don't want to overload on memory
    for i in xrange(len(fz)):
        output[i] = scipy.sum((val - valstar)*M[:,i])
    return output
               
    # due to the larger size of the fz (1000 parameters) in comparison to the parabola (3), I need to generate jacobians

def objectiveJac2(fz, val, c_W, M):
    output = scipy.zeros(fz.shape)
    valstar = c_W*scipy.dot(M, fz)
    output = scipy.dot(valstar - val,M)
    return output
               
    # due to the larger size of the fz (1000 parameters) in comparison to the parabola (3), I need to generate jacobians


def objectiveLogJac(fz, val, c_W, M):
    output = scipy.zeros(fz.shape)
    temp = c_W*scipy.dot(M, fz)
    
    idx = temp != 0

    # I am going to do this in a for loop, because M is huge and I don't want to overload on memory
    for i in xrange(len(fz)):
        output[i] = scipy.sum((scipy.log(temp[idx]) - scipy.log(val[idx]))*M[idx,i]/temp[idx])
    return output

def objectiveLogJac2(fz, val, c_W, M):
    output = scipy.zeros(fz.shape)
    temp = c_W*scipy.dot(M, fz)
    
    idx = temp != 0

    # I am going to do this in a for loop, because M is huge and I don't want to overload on memory
    output = scipy.dot((scipy.log(temp[idx]) - scipy.log(val[idx]))/temp[idx],M[idx])
    return output

############################################################
#                 regularization wrappers                  #
############################################################

# These functions wrap the defined optimizations functions above with regularizations (not standard elastic net style
# but tailored to the problem of fitting the fractional abundances observed (added inputs lam == lambda, or regularization
# parameter, and mat the matrix which does the necessary operation of the regularization

def regFullObj(fz, val, c_W, M, lam, mat):
    output = fullObjective(fz, val, c_W, M)/len(val)

    #regularizer maintaining positive solutions
    reg = dotPositive(mat, fz)
    output += lam*scipy.sum(pow(reg,2))/2./len(val)

    return output

def regFullObjLog(fz, val, c_W, M, lam, mat):
    output = fullObjectiveLog(fz, val, c_W, M)/len(val)
    
    #regularizer maintaining positive solutions
    reg = dotPositive(mat, fz)
    output += lam*scipy.sum(pow(reg,2))/2./len(val)

    return output

def regFullObjJac(fz, val, c_W, M, lam, mat):
    output = objectiveJac(fz, val, c_W, M)/len(val)

    #regularizer maintaining positive solutions
    
    reg = dotPositive(mat, fz)
    output += lam*reg/len(val)

    # I feel like I am missing a minus sign or something
    # I am going to double-check coursera
    
    return output

def regFullObjLogJac(fz, val, c_W, M, lam, mat):
    output = objectiveLogJac(fz, val, c_W, M)/len(val)

    #regularizer maintaining positive solutions
    reg = dotPositive(mat, fz)
    output += lam*reg/len(val)
    
    return output

############################################################
#                  data loading scripts                    #
############################################################

def loadData(name, database='SIFdataMLE2015.db'):
    """ loads data from the fitted database of CSXR data"""
    a = sqlite3.connect(database)
    b = a.cursor()
    return scipy.squeeze(b.execute('SELECT '+name+' FROM shots2015').fetchall())

def loadM(idx, database='SIFweight2015.db'):
    a = sqlite3.connect(database)
    b = a.cursor()
    temp = scipy.squeeze(b.execute('SELECT * FROM shots2015').fetchall())
    order = scipy.searchsorted(temp[:,0],idx)
    return temp[order,1:]

def gen2Diff(matrix):
    """generates a 2nd difference matrix needed for regularization"""
    val = matrix.shape[0]
    output = scipy.eye(val,k=-1) + scipy.eye(val,k=1) - 2*scipy.eye(val)
    return output

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

def conditionVal2(name='val7', nozero=False, conc=None):
    #this is a program which extracts the useful data from the CSXR data

    val = loadData(name)
    idx = loadData('id')
    c_w_l = loadData('c_w_l')
    c_w_qc = loadData('c_w')
    shot = loadData('shot')

    base = loadData('baseline')
    width = loadData('width'+name[-1])

    fiducial = (base+val/scipy.sqrt(scipy.pi)/width)*123+1000

    #c_w_l must have a value below 1e-2 and nonzero
    good = scipy.logical_and(scipy.logical_and(scipy.logical_and(fiducial < 6e4, c_w_l < 1e-2), val*disp > 1e0), c_w_l > 0)
    good2 = scipy.logical_and(scipy.logical_and(scipy.logical_and(fiducial < 6e4, c_w_qc < 1e-2), val*disp > 1e0), c_w_qc > 0)
    goodin = scipy.logical_or(good, good2)

    idx = idx[goodin]
    val = val[goodin]
    shot = shot[goodin]
    c_w_l[scipy.logical_and(scipy.logical_not(good), good2)] = c_w_qc[scipy.logical_and(scipy.logical_not(good), good2)] 
    c_w_l = c_w_l[goodin]

    #val[val < 1.] = 0.

    #normalize to time
    #all valid data with the new computer has 8ms exposure time
    #older shots had 5ms exposure times

    val[shot <= 32858] /= 5e-3
    val[shot > 32858] /= 8e-3

    if nozero:
        good = val != 0
        idx = idx[goodin]
        val = val[goodin]
        c_w_l = c_w_l[goodin]

    
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

    base = loadData('baseline')
    width = loadData('width'+name[-1])

    fiducial = (base+val/scipy.sqrt(scipy.pi)/width)*123+1000

    #c_w_l must have a value below 1e-2 and nonzero
    good = scipy.logical_and(scipy.logical_and(scipy.logical_and(fiducial < 6e4, c_w_l < 1e-2), val*disp > 1e0), c_w_l > 0)

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

    
    return val, c_w_l, idx, good


def modelFunc(cs0, Te1=1e3, Te2=5e4, PEC=None, dataset = 'W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave',loc='data_to_Ian.txt'):
              
    data = scipy.io.readsav(dataset)

    idx = scipy.logical_and(data['en'] > Te1, data['en'] < Te2)
    output = scipy.zeros(data['en'].shape)
    output2 = scipy.zeros(data['en'].shape)
    temp = scipy.arange(len(output))[idx]

    if PEC: 
        inp = adsre.lineInfo(loc)
        PEC = inp(data['en'],PEC)
    else:
        PEC = scipy.ones(data['en'].shape)

    for i in temp:
        output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,cs0]*PEC)

    return output/output.max()


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


def loadVariance(val, c_w_l, M, loc='/afs/ipp-garching.mpg.de/home/i/ianf/python/csxr/lamFitsEye50.p', lims=[2.2e3,9e3], lims2=[2.3e3,7e3]):
    dataset = 'W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave'
    Te = scipy.io.readsav(dataset)['en']

    idx = scipy.logical_and(Te > lims[0], Te < lims[1])
    idxin = scipy.logical_and(Te[idx] > lims2[0], Te[idx] < lims2[1])
    idx = scipy.logical_and(Te > lims2[0], Te < lims2[1])
    data = pickle.load(open(loc,'rb'))


    x = scipy.zeros((len(data),))
    output = scipy.zeros((len(data),))

    for i in xrange(len(output)):
        print(i)
        temp = c_w_l*scipy.dot(M[:,idx], data[i][1].x[idxin])
        
        idx2 = temp != 0
        x[i] = data[i][0]
        output[i] = scipy.var(scipy.log(temp[idx2]) - scipy.log(val[idx2]))


    return x,output
    


############################################################
#                  Fitting data routines                   #
############################################################


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


def fitDataLog2(val, c_w_l, M, nozero=True):

    tikh = sklearn.linear_model.RidgeCV()

    data = val*(4*scipy.pi/c_w_l)
    
    print('fit Ridge with cross-validation')

    p = tikh.fit(scipy.log(M),scipy.log(data)) #do it in log-log space where I can reasonably assume it is a parabola
    
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

def fitDataBFGSM(M, val, c_w_l, init=None, nozero=True, k=3e34, lam=1., name='W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave'): #init is the three initial values of the gaussian needed to fit the data
    """ function for determining the optimal fit given the desired parabolic regularization"""

    #intialize start position
    temp = scipy.io.readsav(name)
    init = temp['abundance'][:,36]
    reg = gen2Diff(init)

    bndarray = scipy.ones((len(init),2))
    bndarray[:,0] = 1e-10
    bndarray[:,1] = 1e10

                        
    Te = temp['en']
    y = time.time()
    output = scipy.optimize.minimize(regFullObjLog,
                                     init,
                                     args=(val, c_w_l, M/k, lam, reg),
                                     #jac=regFullObjLogJac,
                                     bounds=bndarray)
    print(time.time()-y)
    return output

def fitDataBFGSM2(M, val, c_w_l, init=None, nozero=True, k=3e34, lam=1., name='W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave'): #init is the three initial values of the gaussian needed to fit the data
    """ function for determining the optimal fit given the desired parabolic regularization"""

    #intialize start position
    temp = scipy.io.readsav(name)
    init = temp['abundance'][:,36]
    reg = gen2Diff(init)

    bndarray = scipy.ones((len(init),2))
    bndarray[:,0] = 1e-10
    bndarray[:,1] = 1e10

                        
    Te = temp['en']
    y = time.time()
    output = scipy.optimize.minimize(fullObjectiveLog,
                                     init,
                                     args=(val, c_w_l, M/k),
                                     jac=objectiveLogJac2,
                                     bounds=bndarray)
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



############################################################
#                    Plotting routines                     #
############################################################              


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

    if True:
        alpha=.8
        idx = scipy.logical_and(data['en'] > 1e3, data['en'] < 5e4)
        output = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        for i in temp:
            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,46])

        plt.loglog(data['en'],output/output.max(),lw=3,color='crimson',alpha=alpha,linestyle='--',label='W$^{46+}$')   


        output = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        for i in temp:
            
            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,45])

        plt.loglog(data['en'],output/output.max(),lw=3,color='magenta',alpha=alpha,linestyle='--',label='W$^{45+}$')


        output = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        for i in temp:

            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,44])

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

    if True:
        alpha=.8
        idx = scipy.logical_and(data['en'] > 1e3, data['en'] < 5e4)
        output = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        for i in temp:
            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,46])

        plt.loglog(data['en'],output/output.max(),lw=3,color='crimson',alpha=alpha,linestyle='--',label='W$^{46+}$')   

        output = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        for i in temp:
            
            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,45])

        plt.loglog(data['en'],output/output.max(),lw=3,color='magenta',alpha=alpha,linestyle='--',label='W$^{45+}$')


        output = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        for i in temp:

            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,44])

        plt.loglog(data['en'],output/output.max(),lw=3,color='cyan',alpha=alpha,linestyle='--',label='W$^{44+}$')
        plt.xlabel(r'$T_e$ [eV]')
        plt.ylabel(r'const$\cdot I_{CSXR} / c_W \sum M$')

    colorbar = plt.colorbar()
    colorbar.ax.set_ylabel('datapoints in bin')
    plt.legend(loc=4,fontsize=20)
    plt.subplots_adjust(bottom=.12,right=1.)


def visualize5(M, name, nozero=False,k = 2e29,useall=False,lim=1e-2):

    if useall:
        val, c_w_l, idx = conditionVal2(name=name, nozero=nozero)
    else:
        val, c_w_l, idx = conditionVal(name=name, nozero=nozero)
              
    data = scipy.io.readsav('W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave')
    y = time.time()

    maxT = HighTe(M)
    xax = scipy.logspace(2,5,201)
    yax = scipy.logspace(-5,1,101)
    xtemp = xax[1:]/2.+xax[:-1]/2.
    ytemp = yax[1:]/2.+yax[:-1]/2.

    Y,X = scipy.meshgrid(xtemp, ytemp)
    histdata, xed, yed = scipy.histogram2d(maxT,val/c_w_l/scipy.sum(M,axis=1)*k, bins=[xax,yax])
    extent = [xed[0], xed[-1], yed[0], yed[-1]]
    plt.pcolormesh(xtemp,ytemp,histdata.T,norm=LogNorm(), cmap='viridis', rasterized=True)
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
            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,46])

        plt.loglog(data['en'],output/output.max(),lw=3,color='crimson',alpha=alpha,linestyle='--',label='W$^{46+}$')   

        output = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        for i in temp:
            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,45])

        plt.loglog(data['en'],output/output.max(),lw=3,color='magenta',alpha=alpha,linestyle='--',label='W$^{45+}$')
       
        output = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        for i in temp:
            para = fitPuetti(46,lim=lim)
            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*(lognorm(data['en'],para)))

        plt.loglog(data['en'],output/output.max(),lw=3,color='crimson',alpha=alpha,linestyle='-',label='$^*$W$^{46+}$')

        output = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        for i in temp:
            para = fitPuetti(45,lim=lim)
            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*(lognorm(data['en'],para)))

        plt.loglog(data['en'],output/output.max(),lw=3,color='magenta',alpha=alpha,linestyle='-',label='$^*$W$^{45+}$')


        plt.xlabel(r'$T_e$ [eV]')
        plt.ylabel(r'const$\cdot I_{CSXR} / c_W \sum M$')

    colorbar = plt.colorbar()
    colorbar.ax.set_ylabel('datapoints in bin')
    plt.legend(loc=4,fontsize=20)
    plt.subplots_adjust(bottom=.12,right=1.)


def visualize5PEC(M, name, nozero=False,k = 2e29,useall=False,lim=1e-2,loc='data_to_Ian.txt'):

    if useall:
        val, c_w_l, idx = conditionVal2(name=name, nozero=nozero)
    else:
        val, c_w_l, idx = conditionVal(name=name, nozero=nozero)
              
    data = scipy.io.readsav('W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave')
    y = time.time()

    maxT = HighTe(M)

    xax = scipy.logspace(2,5,201)
    yax = scipy.logspace(-5,1,101)
    xtemp = xax[1:]/2.+xax[:-1]/2.
    ytemp = yax[1:]/2.+yax[:-1]/2.

    Y,X = scipy.meshgrid(xtemp, ytemp)
    histdata, xed, yed = scipy.histogram2d(maxT,val/c_w_l/scipy.sum(M,axis=1)*k, bins=[xax,yax])
    extent = [xed[0], xed[-1], yed[0], yed[-1]]
    plt.pcolormesh(xtemp,ytemp,histdata.T,norm=LogNorm(), cmap='viridis', rasterized=True)
    plt.gca().set_yscale('log')
    plt.gca().set_xscale('log')
    plt.gca().set_ylim(1e-4,1e1)
    plt.gca().set_xlim(1e3,2e4)
    
    if True:
        inp = adsre.lineInfo(loc)
        alpha=.8
        idx = scipy.logical_and(data['en'] > 1e3, data['en'] < 5e4)
        output = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        PEC = inp(data['en'], 0)
        for i in temp:
            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,46]*PEC)

        plt.loglog(data['en'],output/output.max(),lw=3,color='crimson',alpha=alpha,linestyle='--',label='W$^{46+}$')   


        output = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        PEC = inp(data['en'], 3)
        for i in temp:
            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,45]*PEC)

        plt.loglog(data['en'],output/output.max(),lw=3,color='magenta',alpha=alpha,linestyle='--',label='W$^{45+}$')
       

        output = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        for i in temp:
            para = fitPuetti(46,lim=lim)
            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*(lognorm(data['en'],para)))

        plt.loglog(data['en'],output/output.max(),lw=3,color='crimson',alpha=alpha,linestyle='-',label='$^*$W$^{46+}$')

        output = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        for i in temp:


            para = fitPuetti(45,lim=lim)
            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*(lognorm(data['en'],para)))

        plt.loglog(data['en'],output/output.max(),lw=3,color='magenta',alpha=alpha,linestyle='-',label='$^*$W$^{45+}$')


        plt.xlabel(r'$T_e$ [eV]')
        plt.ylabel(r'const$\cdot I_{CSXR} / c_W \sum M$')

    colorbar = plt.colorbar()
    colorbar.ax.set_ylabel('datapoints in bin')
    plt.legend(loc=4,fontsize=20,title=r'\underline{\hspace{.5em} \emph{with PEC} \hspace{.5em}}')
    plt.subplots_adjust(bottom=.12,right=1.)

def fitPuettiPlot(inp=46,inp2=45,dataset = 'W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave', lim=1e-2,loc=8):
    data = scipy.io.readsav(dataset)

    Te = data['en']
    lnTe = scipy.log(Te)
    abund = data['abundance'][:,inp]


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
            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,W])

        plt.loglog(data['en'],output/output.max(),lw=3,color=color,alpha=alpha,linestyle='--',label=r'W$^{'+str(W)+'+}$')   

        plt.xlabel(r'$T_e$ [eV]')

    plt.legend(loc=4,fontsize=20)
    plt.subplots_adjust(wspace=0.,bottom=.12,right=1.)

    return colormesh

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
    xax = scipy.logspace(2,5,201)
    yax = scipy.logspace(-5,1,101)
    xtemp = xax[1:]/2.+xax[:-1]/2.
    ytemp = yax[1:]/2.+yax[:-1]/2.

    Y,X = scipy.meshgrid(xtemp, ytemp)
    histdata, xed, yed = scipy.histogram2d(maxT,val/c_w_l/scipy.sum(M,axis=1)*k,bins=[xax,yax])
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
    colorbar.ax.set_ylabel('datapoints in bin')
    leg = plt.legend(loc=4,fontsize=20,title=r'\underline{\hspace{1em} \emph{with PEC} \hspace{1em}}')
    plt.setp(leg.get_title(),fontsize=14)
    plt.subplots_adjust(bottom=.12,right=1.)

def visualize2PEC3(M, name, nozero=False, k=2.636e29, useall=False, lim=1e-2, conc='c_w_l',loc='data_to_Ian.txt'):
    """ goodness of fit vs charge state fractional abundance with expected PEC """

    if useall:
        val, c_w_l, idx = conditionVal2(name=name, nozero=nozero, conc=conc)
    else:
        val, c_w_l, idx = conditionVal(name=name, nozero=nozero, conc=conc)
              
    data = scipy.io.readsav('W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave')
    y = time.time()

    maxT = HighTe(M)
    xax = scipy.logspace(2,5,201)
    yax = scipy.logspace(-5,1,101)

    expdata = val/c_w_l/scipy.sum(M,axis=1)
    histdata, xed, yed = scipy.histogram2d(maxT,expdata*k,bins=[xax,yax])

 
    inp = adsre.lineInfo(loc)
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
        
        if not noPEC:
            rsquared[j-1]= 1- scipy.var(differ)/scipy.var(scipy.log(expdata[idx12]))       
        rsquared2[j-1]= 1- scipy.var(differ2)/scipy.var(scipy.log(expdata[idx22]))
        
    print(rsquared)
    print(rsquared2)

    plt.plot(chargestates,rsquared,color='#A02C2C',marker='s',markersize=10,lw=0,label=r'PEC$\cdot f_Z$')
    plt.plot(chargestates,rsquared2,color='#228B22',marker='o',markersize=10,lw=0,label=r'$ f_Z$')
    plt.xlim([40,50])
    plt.xticks(scipy.mgrid[40:51:1])
    plt.ylim([0,1])

    plt.xlabel(r'charge state')
    plt.ylabel(r'$R^2$ of $\log$-$\log$ fit')
    leg = plt.legend(numpoints=1,fontsize=18,title=r'\underline{\hspace{.5ex} max$(R^2)=.564$ \hspace{.5ex}}')
    plt.setp(leg.get_title(),fontsize=14)
    plt.subplots_adjust(bottom=.2)
    plt.show()

    return output2[p],expdata

def visualize2PEC4(M, name, nozero=False, useall=False, lim=1e-2, conc='c_w_l',loc='data_to_Ian.txt'):
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

    expdata = val/c_w_l/scipy.sum(M, axis=1)

    plt.gca().set_xscale('log')
    plt.gca().set_xlim(1e3,2e4)
   
    inp = adsre.lineInfo(loc)

    pecDict = {43:7,44:6,45:3,46:0,47:8} #related to line PEC document
    variances =  scipy.zeros(data['en'].shape)

    if True:

        j = 45
        idx = scipy.logical_and(data['en'] > 1e3, data['en'] < 5e4)
        alpha = .8
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
            #find index of 
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
  
    expdata = val/c_w_l/scipy.sum(M,axis=1)*k

    plt.scatter(maxT,expdata,c=c_w_l,cmap='viridis', rasterized=True, norm=LogNorm(),alpha=.25)
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
    colorbar.ax.set_ylabel('$c_W$')
    leg = plt.legend(loc=4,fontsize=20,title=r'\underline{\hspace{1em} \emph{with PEC} \hspace{1em}}')
    plt.setp(leg.get_title(),fontsize=14)
    plt.subplots_adjust(bottom=.12,right=1.)


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

    plt.pcolormesh(xax, yax, histdata.T, cmap='viridis', rasterized=True, norm=LogNorm(),vmin=1e38)
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
        #plt.loglog(data['en'],output/output.max(),lw=3,color='magenta',alpha=alpha,linestyle='-',label=r'_nolegend')

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
    colorbar.ax.set_ylabel('$\sum M$ [m$^{-5}$]')
    leg = plt.legend(loc=4,fontsize=20,title=r'\underline{\hspace{1em} \emph{with PEC} \hspace{1em}}')
    plt.setp(leg.get_title(),fontsize=14)
    plt.subplots_adjust(bottom=.12,right=.99)

    return histdata

def visualizeCoVar(M, name, covar, nozero=False, k=2.5e29, lim=1e-2, conc='c_w_l',loc='data_to_Ian.txt'):
    """ sort the data like the PEC graphs, and then calculate parameters of interest"""
 
    val, c_w_l, idx, good = conditionVal4(name=name, nozero=nozero, conc=conc)
    val4 = loadData(covar)[good]
              
    data = scipy.io.readsav('W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave')
    y = time.time()

    maxT = HighTe(M)
    val4 = scipy.sum(M,axis=1)

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
    colorbar.ax.set_ylabel('$\sum M$ [m$^{-5}$]')
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
    xax = scipy.logspace(2,5,201)
    yax = scipy.logspace(-5,1,101)
    xtemp = xax[1:]/2.+xax[:-1]/2.
    ytemp = yax[1:]/2.+yax[:-1]/2.

    Y,X = scipy.meshgrid(xtemp, ytemp)
    histdata, xed, yed = scipy.histogram2d(maxT[good2],val[good2]/c_w_l[good2]/scipy.sum(M[good2],axis=1)*k,bins=[xax,yax])
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

def calibk(M, name, nozero=False, useall=False, lim=1e-2, conc='c_w_l',loc='data_to_Ian.txt'):

    if useall:
        val, c_w_l, idx = conditionVal2(name=name, nozero=nozero, conc=conc)
    else:
        val, c_w_l, idx = conditionVal(name=name, nozero=nozero, conc=conc)
              
    data = scipy.io.readsav('W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave')
    y = time.time()


    bins = scipy.logspace(34.,37.,1e3+1)
   
    maxT = HighTe(M)
    sumM = scipy.sum(M,axis=1)

    tedata = scipy.searchsorted(data['en'],maxT)
    yvals = val/c_w_l/sumM
    idxin = scipy.isfinite(yvals) 
    inp = adsre.lineInfo(loc)

    if True:
        alpha=.8
        idx = scipy.logical_and(data['en'] > 1e3, data['en'] < 5e4)
        output = scipy.zeros(data['en'].shape)
        output2 = scipy.zeros(data['en'].shape)

        temp = scipy.arange(len(output))[idx]
        PEC = inp(data['en'], 0)
        for i in temp:
            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,46]*PEC)
            output2[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,46])

        output /=output.max()
        output2 /=output2.max()

        tester = (scipy.log(output[tedata])-scipy.log(yvals))/scipy.log(10)
        idxin = scipy.isfinite(tester)

        tester2 = (scipy.log(output2[tedata])-scipy.log(yvals))/scipy.log(10)
        idxin2 = scipy.isfinite(tester2)

        histdata, tempbin = scipy.histogram(pow(10,tester[idxin]),bins=bins)
        histdata2, tempbin2 = scipy.histogram(pow(10,tester2[idxin2]),bins=bins)


        print(46,1-scipy.var(tester[idxin])/scipy.var(scipy.log(yvals[idxin])/scipy.log(10)),1-scipy.var(tester2[idxin2])/scipy.var(scipy.log(yvals[idxin2])/scipy.log(10)))
        
        plt.loglog((tempbin[1:]+tempbin[:-1])/2.,histdata,'b')
        plt.loglog((tempbin2[1:]+tempbin2[:-1])/2.,histdata2,'b--')


        output = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        PEC = inp(data['en'], 3)
        for i in temp:
            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,45]*PEC)
            output2[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,45])

        output /=output.max()
        output2 /=output2.max()

        tester = (scipy.log(output[tedata])-scipy.log(yvals))/scipy.log(10)
        idxin = scipy.isfinite(tester)

        tester2 = (scipy.log(output2[tedata])-scipy.log(yvals))/scipy.log(10)
        idxin2 = scipy.isfinite(tester2)

        histdata, tempbin = scipy.histogram(pow(10,tester[idxin]),bins=bins)
        histdata2, tempbin2 = scipy.histogram(pow(10,tester2[idxin2]),bins=bins)


        print(45,1-scipy.var(tester[idxin])/scipy.var(scipy.log(yvals[idxin])/scipy.log(10)),1-scipy.var(tester2[idxin2])/scipy.var(scipy.log(yvals[idxin2])/scipy.log(10)))
        
        plt.loglog((tempbin[1:]+tempbin[:-1])/2.,histdata,'g')
        plt.loglog((tempbin2[1:]+tempbin2[:-1])/2.,histdata2,'g--')

        output = scipy.zeros(data['en'].shape)
        temp = scipy.arange(len(output))[idx]
        PEC = inp(data['en'], 6)
        for i in temp:

            output[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,44]*PEC)
            output2[i] = scipy.sum(testM.normMn2(10, scipy.log(data['en']),  scipy.log(5e2), scipy.log(data['en'][i]), .125)*data['abundance'][:,44])
        
        output /=output.max()
        output2 /= output2.max()

        tester = (scipy.log(output[tedata])-scipy.log(yvals))/scipy.log(10)
        idxin = scipy.isfinite(tester)

        tester2 = (scipy.log(output2[tedata])-scipy.log(yvals))/scipy.log(10)
        idxin2 = scipy.isfinite(tester2)

        histdata, tempbin = scipy.histogram(pow(10,tester[idxin]),bins=bins)
        histdata2, tempbin2 = scipy.histogram(pow(10,tester2[idxin2]),bins=bins)

        print(44,1-scipy.var(tester[idxin])/scipy.var(scipy.log(yvals[idxin])/scipy.log(10)),1-scipy.var(tester2[idxin2])/scipy.var(scipy.log(yvals[idxin2])/scipy.log(10)))
        
        plt.loglog((tempbin[1:]+tempbin[:-1])/2.,histdata,'r')
        plt.loglog((tempbin2[1:]+tempbin2[:-1])/2.,histdata2,'r--')

        plt.xlabel(r'$T_e$ [eV]')


def visualize2Ratio(M, name, name2, nozero=False, k=1., useall=False, lim=1e-2, conc='c_w_l',loc='data_to_Ian.txt'):
    """ log plot of the data with only PEC data"""

    val = loadData(name)
    val2 = loadData(name2)
    cw = loadData('c_w_l')
    cw2 = loadData('c_w')
    idxin = scipy.logical_or(cw > 0, cw2 >0)
    idxin2 = scipy.logical_and(val[idxin]*disp > 1e2, val2[idxin]*disp > 1e2)

    idx = loadData('id')

    val = val2/val
    val = val[idxin]

    data = scipy.io.readsav('W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave')
    y = time.time()

    maxT = HighTe(M)
    xax = scipy.logspace(2,5,201)
    yax = scipy.logspace(-4,2,101)
    xtemp = xax[1:]/2.+xax[:-1]/2.
    ytemp = yax[1:]/2.+yax[:-1]/2.

    print(maxT.shape,val.shape)

    Y,X = scipy.meshgrid(xtemp, ytemp)
    histdata, xed, yed = scipy.histogram2d(maxT[idxin2],val[idxin2]*k,bins=[xax,yax])
    extent = [xed[0], xed[-1], yed[0], yed[-1]]
    plt.pcolormesh(xax,yax,histdata.T, cmap='viridis', rasterized=True,vmin=1.,norm=LogNorm())
    plt.gca().set_yscale('log')
    plt.gca().set_xscale('log')
    plt.gca().set_ylim(1e-2,1e2)
    plt.gca().set_xlim(1e3,2e4)
   
    cmap = plt.cm.get_cmap('viridis')
    cmap.set_under('white')
 
    inp = adsre.lineInfo(loc)


    if True:

        data1 = modelFunc(44,PEC=6)
        data2 = modelFunc(45,PEC=3)
        data3 = modelFunc(46,PEC=0)

        data2 = modelFunc(45)
        data3 = modelFunc(46)
        #plt.loglog(data['en'],data2/data3)
        plt.axhline(1., lw=2.,linestyle='--',color='crimson')
        plt.text(8e3, 2e-2, '1: '+str(semiEmperical[int(name[-1])])+' nm')
        plt.text(8e3, 1.2e-2, '2: '+str(semiEmperical[int(name2[-1])])+' nm')

        plt.xlabel(r'$T_e$ [eV]')
        plt.ylabel(r'const$\cdot I_{CSXR2} / I_{CSXR1}$')

    colorbar = plt.colorbar()
    colorbar.ax.set_ylabel('datapoints in bin')
    #leg = plt.legend(loc=4,fontsize=20,title=r'\underline{\hspace{1em} \emph{with PEC} \hspace{1em}}')
    #plt.setp(leg.get_title(),fontsize=14)
    plt.subplots_adjust(bottom=.12,right=1.)


def plotCV(data,lam=[0,6],pts=1001):
    lamin = scipy.logspace(lam[0],lam[1],pts)

    plt.semilogx(lamin, data,'.', color='#0E525A',markersize=10)
    plt.xlabel('$\lambda$')
    plt.ylabel('var$(\ln(I_{CSXR}) - \ln(c_W M \cdot f_{Z,fit}))$')
    plt.show()


def regSolutionExample(minval, loc='/afs/ipp-garching.mpg.de/home/i/ianf/python/csxr/lamFitsEye50.p', lims=[2.2e3,9e3], lims2=[2.3e3,7e3],alpha=.8):

    dataset = scipy.io.readsav('W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave')    
    Te = dataset['en']

    idx = scipy.logical_and(Te > lims[0], Te < lims[1])
    idx2 = scipy.logical_and(Te[idx] > lims2[0], Te[idx] < lims2[1])
    idx = scipy.logical_and(Te > lims2[0], Te < lims2[1])
    data = pickle.load(open(loc,'rb')) 
    
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

    plt.loglog(Te[idx],data[0][1].x[idx2]/data[0][1].x[idx2].max(),color='#228B22',lw=3,label='$\lambda=0$')
    plt.loglog(Te[idx],data[minval][1].x[idx2]/data[minval][1].x[idx2].max(),color='#0E525A',lw=3,label='CV $\lambda$')

    plt.loglog(Te,data3,lw=3,color='crimson',alpha=alpha,linestyle='--',label=r'W$^{46+}$')
    plt.loglog(Te,data2,lw=3,color='magenta',alpha=alpha,linestyle='--',label=r'W$^{45+}$')
    plt.loglog(Te,data1,lw=3,color='cyan',alpha=alpha,linestyle='--',label=r'W$^{44+}$')

    plt.xlim(1e3,2e4)
    plt.ylim(1e-2,1e1)
    plt.xlabel('$T_e$ [eV]')
    plt.ylabel('PEC$\cdot f_Z/$max(PEC$\cdot f_Z$)')
    leg = plt.legend(fontsize=20,title=r'\underline{\hspace{.5em} \emph{solid} - \emph{$f_Z$ fits}\hspace{.5em} }')   
    plt.setp(leg.get_title(),fontsize=14)
    plt.subplots_adjust(bottom=.12)
    plt.show()

def regSolutionExample2(minval, loc='/afs/ipp-garching.mpg.de/home/i/ianf/python/csxr/lamFitsDiff36.p', lims=[2.2e3,9e3], lims2=[2.3e3,7e3],alpha=.8):

    data4 = pickle.load(open(loc,'rb')) 

    loc = '/afs/ipp-garching.mpg.de/home/i/ianf/python/csxr/lamFitsEye50.p'
    dataset = scipy.io.readsav('W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave')    
    Te = dataset['en']

    idx = scipy.logical_and(Te > lims[0], Te < lims[1])
    idx2 = scipy.logical_and(Te[idx] > lims2[0], Te[idx] < lims2[1])
    idx = scipy.logical_and(Te > lims2[0], Te < lims2[1])
    data = pickle.load(open(loc,'rb')) 
    
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

    plt.loglog(Te[idx],data[0][1].x[idx2]/data[0][1].x[idx2].max(),color='#228B22',lw=3,label='$\lambda=0$')
    plt.loglog(Te[idx],data4[minval][1].x[idx2]/data4[minval][1].x[idx2].max(),color='gold',lw=3,label='$\partial^2 f_Z$ $\lambda$')
    plt.loglog(Te[idx],data[603][1].x[idx2]/data[603][1].x[idx2].max(),color='#0E525A',lw=3,label='CV $\lambda$')

    plt.loglog(Te,data3,lw=3,color='crimson',alpha=alpha,linestyle='--',label=r'W$^{46+}$')
    plt.loglog(Te,data2,lw=3,color='magenta',alpha=alpha,linestyle='--',label=r'W$^{45+}$')
    plt.loglog(Te,data1,lw=3,color='cyan',alpha=alpha,linestyle='--',label=r'W$^{44+}$')

    plt.xlim(1e3,2e4)
    plt.ylim(1e-2,1e1)
    plt.xlabel('$T_e$ [eV]')
    plt.ylabel('PEC$\cdot f_Z/$max(PEC$\cdot f_Z$)')
    leg = plt.legend(fontsize=20,title=r'\underline{\hspace{.5em} \emph{solid} - \emph{$f_Z$ fits}\hspace{.5em} }')   
    plt.setp(leg.get_title(),fontsize=14)
    plt.subplots_adjust(bottom=.12)
    plt.show()
