import scipy
import dd
#import sklearn
import scipy.optimize
from scipy.optimize.optimize import _approx_fprime_helper
import _gauss as _gauss2 #custom extension to try and speed up minimize function

ArHe_like_lambda = {'w':0.39492, 'x':0.3966, 'y':0.39695, 'z':0.39943} #[nm]
ArLi_like_lambda = {'a':0.39852, 'j':0.39941, 'k':0.399, 'q':0.39815, 'r':0.39836, 's':0.39678, 't':0.39687} #[nm]
Wlambda = {'px375':0.39647953} #[nm]


lines = {'S':{14:[.39039,.39219,.39501,.39978,.39988],15:[.3990802,.3991944]},
'Mo':{39:[.3972630],40:[.393127,.3946071]},
'Kr':{33:[.39766,.39870],34:[.392229,.392582,.394582,.39515,.396901]},
'W':{43:[.39636],44:[.39097,.3973,.39895],45:[.39329,.3933]},
'Ar':{14:[.39998],15:[.396159,.396561,.39676,.39685,.39813,.398167,.39834,.398572,.398573,.398979,.398981,.39938,.399388],16:[.39490665,.39693556,.39658675,.39941451]}}

#                            ~W44+         ~W45+         W-line       ~W43+ ???   X-line Y-line ~W44+ ~W44+  Z-line
semiEmperical = scipy.array([.3909,.39316,.39331,.39374,.39492,.3955,.3963,.39648,.3966,.39695,.3975,.39887,.39943])

#2017 calibration off of 34228 w line
calib = 33636.23 #33354.6# 33636.23
disp = 86363.6363 #85650.6# 86363.6363

# internal fitting bounds and initial value for baseline and gaussian height, offset, and width
_bounds = scipy.array([[1,65536],[0,1e15],[-3e-5,3e-5],[5e-5,3e-4]])
_bounds2 = scipy.array([[0,scipy.inf],[0,scipy.inf],[-3e-5,3e-5],[0,scipy.inf]])

_initial = scipy.array([0.,1.,1.,.7e-4,0.])

#class SpecData(object):
#
#    def __init__(self, filename="SIF", data="DataSIF"):
#        self._shotfile = None
#        self._data = None
#
#        self._filename = filename
#        self._dataname = data
#
#    def __call__(shot):
#        #DATA GET
#        self._shotfile = dd.shotfile(self._filename)
#        self._data = shotfile(self._dataname)


#This yanks data for the analysis (for fitting gaussians)
def SIFData(shot, filename="SIF", data="DataSIF", offset=0, rev=True):

    shotfile = dd.shotfile(filename,shot)
    temp = shotfile(data)
    temp.data = temp.data[:,:1024]
    temp.data = temp.data[:,::-1] #flip axis for wavelength data
    temp.wavelength = (scipy.arange(temp.data.shape[1]) + offset + calib)/disp
    temp.wavelength = temp.wavelength[:1024]
    #temp.wavelength = (scipy.arange(1250) - 34334.5 + offset)/(-85650.6) #give it a wavelength parameter.  This is hardcoded and derived from Sertoli's work in /u/csxr/sif/sif.pro
    return temp #send it out into the wild

#Load example data (in this case the example shot from Marco's thesis (23091), which hopefully has similar shots, but we will begin by trying to fit this data initially.


def _gauss(x, inp):
    #Gaussian function for fitting, dot needed to match proper
    return scipy.sum(inp[0]*scipy.exp(-1*pow((x - scipy.dot(scipy.ones((len(x),1)),
                                                            scipy.atleast_2d(inp[1])).T).T/inp[2],2)),axis=1)

def gaussian(*args):
    x = scipy.array(args[0])
    inp = scipy.squeeze(args[1:])
    # this assembles the addition of multiple gaussians with a baseline (used for fitting the data)
    return inp[0] + _gauss(x,scipy.array(inp[1:]).reshape((3,(len(inp)-1)/3),order='F')) #possibly for fortan order

def compare(inp, xdata, ydata):
    #wrapper to use for in scipy.optimize.minimize, convertings the RMSE in 
    return scipy.sqrt(scipy.sum(pow(gaussian(xdata,inp) - ydata, 2)))

def compare2(inp, xdata, ydata):
    #wrapper to use for in scipy.optimize.minimize, convertings the RMSE in 
    #print(inp[0])
    return scipy.sum(pow(ydata - inp[0]- _gauss2.gauss(xdata,inp[1:]), 2))

def compare3(inp, xdata, ydata):
    #wrapper to use for in scipy.optimize.minimize, convertings the RMSE in 
    #print(inp[0])

    #temp = scipy.sqrt(scipy.sum(pow(ydata - gaussian(xdata,inp), 2)))
    
    #temp2 = compare(inp,xdata,ydata)
    temp3 = interface(inp,xdata,ydata)
    temp2 = _approx_fprime_helper(inp,compare,1e-8,args=(xdata,ydata))
    print(abs(temp3[1:]-temp2)/temp2)
    #j=3
    #sep = inp[j]*1e-6
    #inp[j] += sep
    #temp2 = scipy.sqrt(scipy.sum(pow(ydata - gaussian(xdata,inp), 2)))
    #tempderiv2 = (temp2-temp)/(sep)
    #print((temp3[1][j]-tempderiv2)/tempderiv2)
#print((temp-temp3,temp3-temp2))

    return temp3[0]

def compareln(inp, xdata, ydata):
    #wrapper to use for in scipy.optimize.minimize, convertings the RMSE in 
    #print(inp[0])
    out = inp[0]+ _gauss2.gauss(xdata, inp[1:])
    return scipy.sum(out/123.-ydata/123.*scipy.log(out/123.))

def compareln2(inp, xdata, ydata):
    #wrapper to use for in scipy.optimize.minimize, convertings the RMSE in 
    #print(inp[0])
    out = inp[0]+ _gauss2.gauss2(xdata, inp[1:])
    return scipy.sum(out-ydata*scipy.log(out))

def interface(inp, xdata, ydata):
    p2 = _gauss2.gaussjac(xdata,ydata,inp)
    #p = scipy.zeros((len(inp)+1,))
    #p[0] = compare(inp,xdata,ydata)
    #p[1:] = _approx_fprime_helper(inp,compare,1e-8,args=(xdata,ydata))
    #print('new')
    #print(inp)
    #print(p)
    #print(p2)
    if scipy.any(scipy.isnan(p2)):
        print(inp,p2)
    return p2[0],p2[1:]
    #return compare(inp,xdata,ydata),_approx_fprime_helper(inp,compare,1e-8,args=(xdata,ydata))

def interface2(inp, xdata, ydata):
    p = _gauss2.gaussjac(xdata,ydata,inp)
    return p[1:]

def interface3(inp, xdata, ydata):
    p = _gauss2.gaussjac(xdata,ydata,inp)
    return p[0]

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

def interface5(inp, xdata, ydata):
    #p2 = _gauss2.gpmle(xdata,ydata,inp)
    #p = scipy.zeros((len(inp)+1,))
    p2 = compareln2(inp, xdata, ydata)
    #print(p2-_gauss2.gpmle(xdata,ydata,inp)[0])
    #p[1:] = _approx_fprime_helper(inp,compare,1e-8,args=(xdata,ydata))
    #print('new')
    #print(inp)
    #print(p)
    #print(p2)
    if scipy.any(scipy.isnan(p2)):
        print(inp,p2,'error5')
    return p2#[0],p2[1:]

def _assembleInit(xdata, ydata, bounds=None):

    #INTITIALIZE VARIABLES
    init = scipy.zeros((3*len(semiEmperical)+1,))
    output = scipy.zeros((len(init), 2))
    ones = scipy.ones(semiEmperical.shape)
    
    #set baseline
    axis = scipy.mgrid[-32:65569:64]
    init[0] = (scipy.histogram(ydata,bins=axis)[0]).argmax()*64#ydata.min()#_initial[0]
    output[0] = _bounds[0]

    #set peak values
    init[1::3] = ydata[scipy.searchsorted(xdata, semiEmperical)] - init[0]
    output[1::3] = scipy.array([_bounds[1,0]*ones, _bounds[1,1]*ones]).T

    #set offsets
    init[2::3] = semiEmperical
    output[2::3] = (scipy.array([_bounds[2,0]*ones, _bounds[2,1]*ones]) + semiEmperical).T

    #set width values
    init[3::3] = _initial[3]*ones
    output[3::3] = scipy.array([_bounds[3,0]*ones, _bounds[3,1]*ones]).T

    if not bounds == None:
        return init, output
    else:
        return init

def _assembleInit2(xdata, ydata, bounds=None):

    #INTITIALIZE VARIABLES
    init = scipy.zeros((3*len(semiEmperical)+1,))
    output = scipy.zeros((len(init), 2))
    ones = scipy.ones(semiEmperical.shape)
    
    #set baseline
    axis = scipy.mgrid[-32:65569:64]
    init[0] = (scipy.histogram(ydata,bins=axis)[0]).argmax()*64#ydata.min()#_initial[0]
    output[0] = _bounds[0]

    #set peak values
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

def _fitData(idx,xdata,data,bounds=None,method=None,tol=None):
    ydata =data[idx]

    if not bounds == None:
        inp, bounds = _assembleInit(xdata, ydata, bounds=bounds)
    else:
        inp = _assembleInit(xdata, ydata)
        
    #testdata = gaussian(xdata, inp) 
        
    output = scipy.optimize.minimize(compare2,
                                     inp,
                                     args=(xdata,ydata),
                                     method=method,
                                     bounds=bounds,
                                     tol=tol).x
    
    return output

def _fitData2(idx, xdata, data, bounds=None, method=None, tol=None):

    ydata =data[idx]

    if not bounds == None:
        inp, bounds = _assembleInit(xdata, ydata, bounds=bounds)
        testdata = gaussian(xdata, inp) 
        
        output = scipy.optimize.curve_fit(gaussian,
                                          xdata,
                                          ydata,
                                          inp,
                                          method=method,
                                          bounds=bounds)[0]

    else:
        inp = _assembleInit(xdata, ydata)

        testdata = gaussian(xdata, inp) 
        
        output = scipy.optimize.curve_fit(gaussian,
                                          xdata,
                                          ydata,
                                          inp,
                                          method=method)[0]
    
    return output

def _fitData3(idx, xdata, data, bounds=None, method=None, tol=None):

    ydata =data[idx]

    # need to subtract baseline from data


    if not bounds == None:
        inp, bounds = _assembleInit(xdata, ydata, bounds=bounds)
    else:
        inp = _assembleInit(xdata, ydata)
    #inp[0] *= .9 
        
    output = scipy.optimize.minimize(interface3,
                                     inp,
                                     args=(xdata,ydata),
                                     method=method,
                                     bounds=bounds).x
   
    output2 = output[0] + _gauss2.gauss(xdata,output[1:])

    return output2

def _fitData4(idx, xdata, data, bounds=None, method=None, tol=None):

    ydata =data[idx]
    # need to subtract baseline from data


    if not bounds == None:
        inp, bounds = _assembleInit(xdata, ydata, bounds=bounds)
    else:
        inp = _assembleInit(xdata, ydata)
    #inp[0] *= .9 
        
    output = scipy.optimize.minimize(interface,
                                     inp,
                                     args=(xdata,ydata),
                                     method=method,
                                     bounds=bounds,
                                     tol=tol,
                                     jac=True).x

    output2 = output[0] + _gauss2.gauss(xdata,output[1:])
   
    return output2




def _fitData5(xdata, ydata, bounds=None, method=None, tol=None):

    # need to subtract baseline from data


    if not bounds == None:
        inp, bounds = _assembleInit(xdata, ydata, bounds=bounds)
    else:
        inp = _assembleInit(xdata, ydata)
    #inp[0] *= .9 
        
    output = scipy.optimize.minimize(compare3,
                                     inp,
                                     args=(xdata,ydata),
                                     method=method,
                                     tol=tol,
                                     bounds=bounds).x

    output2 = output[0] + _gauss2.gauss(xdata,output[1:])
   
    return output2

def _fitData6(xdata, ydata, bounds=None, method=None, tol=None):

    # need to subtract baseline from data


    if not bounds == None:
        inp, bounds = _assembleInit(xdata, ydata, bounds=bounds)
    else:
        inp = _assembleInit(xdata, ydata)
    #inp[0] *= .9 
        
    output = scipy.optimize.minimize(interface,
                                     inp,
                                     args=(xdata,ydata),
                                     method=method,
                                     bounds=bounds,
                                     tol=tol,
                                     jac=True).x

    output2 = output[0] + _gauss2.gauss(xdata,output[1:])
   
    return output2

def _fitData7(xdata, ydata, bounds=None, method=None, tol=None):

    # need to subtract baseline from data


    if not bounds == None:
        inp, bounds = _assembleInit(xdata, ydata, bounds=bounds)
    else:
        inp = _assembleInit(xdata, ydata)
    #inp[0] *= .9 
        
    output = scipy.optimize.fmin_l_bfgs_b(interface,
                                          inp,
                                          args=(xdata,ydata),
                                          bounds=bounds)#,
                                          #approx_grad=True)

    #output2 = output[0] + _gauss2.gauss(xdata,output[1:])
   
    return output

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


def _fitData9(idx, xdata, data, bounds=None, method=None, tol=None):

    ydata =data[idx] - 1000.
    # need to subtract baseline from data


    if not bounds == None:
        inp, bounds = _assembleInit2(xdata, ydata, bounds=bounds)
    else:
        inp = _assembleInit2(xdata, ydata)
    #inp[0] *= .9 9 
    inp[0] /= 123.
    inp[1::3] /=123.
        
    output = scipy.optimize.minimize(interface5,
                                     inp,
                                     args=(xdata,ydata/123.),
                                     method=method,
                                     bounds=bounds,
                                     tol=tol).x
    print(output[1::3],'ideal')
    
    output2 = output[0] + _gauss2.gauss2(xdata,output[1:])
   
    return output2*123+1000.
