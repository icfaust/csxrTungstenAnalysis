import scipy
import scipy.interpolate
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import rc

rc('text',usetex=True)
rc('text.latex',preamble='\usepackage{xcolor}')
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
#rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern Sans serif']})
rc('font',size=20)
disp = 86363.6363

# first test will be to use the really rough profiles given my Thomas' code
# first is a direct reimplementation

def te(rho, tcore, tped):
    #horrible function from /u/dlbo/w_diag/fit_function_te_profile_new.pro, used by the GIW shotfile generator w_diag

    #corr = scipy.exp(-3.*pow(.97-rho,2))
    #temp = ((1-corr)+.97*6*scipy.exp(-3*pow(.97,2))*(rho*scipy.exp(-8*pow(rho,2))-.97*scipy.exp(-8*pow(.97,2))))/0.940374146
    temp = (_func(rho)- _func(.97))/ (_func(0.)-_func(.97))
    temp[rho > .97] = 0.
    
    return temp*(tcore-tped)*scipy.exp(-2*pow(rho,2))+tped*.5*(1+scipy.tanh((.985-rho)*60))


def ne(rho, ncore, nped):
    #horrible function from /u/dlbo/w_diag/fit_function_ne_profile.pro, used by the GIW shotfile generator w_diag
    temp = (_func(rho)- _func(.97))/ (_func(0.)-_func(.97))
    temp[rho > .97] = 0.
    
    return temp*(ncore-nped)*scipy.exp(-2*pow(rho,2))+nped*.5*(1+scipy.tanh((.99-rho)*80))


def _func(rho):

    corr = scipy.exp(-3.*pow(.97-rho,2))
    coeff1 = (1-corr) + (scipy.exp(-3*pow(.97,2))*.97*3*2)*rho*scipy.exp(-8.*pow(rho,2))
    return coeff1


def te2rho(Te, tcore, tped):
    #solve for rhos for given Te
    tcore = scipy.atleast_1d(tcore)
    tped = scipy.atleast_1d(tped)
    Te = scipy.atleast_2d(Te)
    idx = len(tcore)

    output = scipy.zeros((idx,len(Te)))
    rho = scipy.linspace(0,1,1e2+1)
    
    for i in xrange(idx):
        tetemp = te(rho,tcore[idx],tped[idx])
        temp = scipy.interpolate.interp1d(tetemp,
                                          rho,
                                          kind='cubic',
                                          bounds_error=False)
        output[i] = temp(Te[i])  # find positions of each temperature
    return output


def te2rho2(Te, tcore, tped):
    #solve for rhos for given Te
    rho = scipy.linspace(0,1,1e4+1)
    tetemp = te(rho,tcore,tped)
    temp = scipy.interpolate.interp1d(tetemp,
                                      rho,
                                      kind='linear',
                                      bounds_error=False)
    output = temp(Te)  # find positions of each temperature
    return output



def plot():
    rho = scipy.linspace(0,1,1e3+1)


    plt.plot(rho,te(rho,1.,.25),color='#228B22',lw=3,label='4')
    plt.plot(rho,te(rho,1.,.5),color='#0E525A',lw=3,label='2')

    plt.xlabel(r'$\rho$')    
    plt.ylabel(r'$T_e/T_{e,core}$')
    leg = plt.legend(fontsize=20,title=r'\underline{$T_{e,core}/T_{e,ped}$}')   
    plt.setp(leg.get_title(),fontsize=18)
    plt.subplots_adjust(bottom=.12)

    plt.show()
