import scipy
import scipy.signal as sig
import scipy.optimize
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import dd
import multiprocessing as mp
import data as SIFdata
import _gauss

ArHe_like_lambda = {'w':0.39492, 'x':0.3966, 'y':0.39695, 'z':0.39943} #[nm]
ArLi_like_lambda = {'a':0.39852, 'j':0.39941, 'k':0.399, 'q':0.39815, 'r':0.39836, 's':0.39678, 't':0.39687} #[nm]
Wlambda = {'px375':0.39647953} #[nm]

laxis = (scipy.arange(1024) - 34334.5)/(-85650.6)


print(ArHe_like_lambda['w'])


class DiscreteSlider(Slider):
    """A matplotlib slider widget with discrete steps."""
    def __init__(self, *args, **kwargs):
        """Identical to Slider.__init__, except for the "increment" kwarg.
        "increment" specifies the step size that the slider will be discritized
        to."""
        self.inc = kwargs.pop('increment', 1.0)
        Slider.__init__(self, *args, **kwargs)

    def set_val(self, val):
        discrete_val = int(val / self.inc) * self.inc
        # We can't just call Slider.set_val(self, discrete_val), because this 
        # will prevent the slider from updating properly (it will get stuck at
        # the first step and not "slide"). Instead, we'll keep track of the
        # the continuous value as self.val and pass in the discrete value to
        # everything else.
        xy = self.poly.xy
        xy[2] = discrete_val, 1
        xy[3] = discrete_val, 0
        self.poly.xy = xy
        self.valtext.set_text(self.valfmt % discrete_val)
        if self.drawon: 
            self.ax.figure.canvas.draw()
        self.val = val
        if not self.eventson: 
            return
        for cid, func in self.observers.iteritems():
            func(discrete_val)


def plotSIF3(shot, init=0, offset=-50):

    #DATA GET
    data = SIFdata.SIFData(shot)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    #plt.axvline(ArHe_like_lambda['w'],color='b')
    #plt.axvline(ArHe_like_lambda['x'],color='b')
    #plt.axvline(ArHe_like_lambda['y'],color='b')
    #plt.axvline(ArHe_like_lambda['z'],color='b')
    #plt.axvline(ArLi_like_lambda['a'],color='r')
    #plt.axvline(ArLi_like_lambda['j'],color='r')
    #plt.axvline(ArLi_like_lambda['q'],color='r')
    #plt.axvline(ArLi_like_lambda['r'],color='r')
    #plt.axvline(ArLi_like_lambda['s'],color='r')
    #plt.axvline(ArLi_like_lambda['t'],color='r')
    plt.axvline(Wlambda['px375'],color='g')
    temp = ['r','g','b','k','c','y']
    inp = 0

    temp2 = [['W',43,'c'],
             ['W',44,'c'],
             ['W',45,'c'],
             ['Ar',16,'b']]
    
    for i in temp2:
        print(i,temp[inp])
        for j in SIFdata.lines[i[0]][i[1]]:
            plt.axvline(j,color=i[2])



    #l2, = plt.semilogy(laxis2+offset/(-85650.6), data.data[init], lw=2, color='green')
    l, = plt.plot(data.wavelength, sig.medfilt(data.data[init],5), lw=2, color='red')
    plt.ylim((0,5e4))#scipy.nanmax(data.data)))


    axcolor = 'lightgoldenrodyellow'
    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])

    sfreq = DiscreteSlider(axfreq, 'Freq', 0, len(data.time), valinit=init, valfmt='%0.0i', increment=1.0)

    def update(val):
        freq = int(sfreq.val)
        print(freq)
        #l2.set_ydata(data.data[val])
        l.set_ydata(sig.medfilt(data.data[val],5))
        fig.canvas.draw_idle()
    sfreq.on_changed(update)
    plt.show()

def _plotSIF(shot, init=0, offset=0):
    semiEmperical = scipy.array([.3909,.39316,.39331,.39374,.39492,.3955,.3963,.39648,.3966,.39695,.3975,.39887,.39943])
    #global ArHe_like_lambda
    #DATA GET
    data = SIFdata.SIFData(shot,offset=offset)
    fig, ax = plt.subplots()
   
    #plt.axvline(ArHe_like_lambda['w'],color='b')
    #plt.axvline(ArHe_like_lambda['x'],color='b')
    #plt.axvline(ArHe_like_lambda['y'],color='b')
    #plt.axvline(ArHe_like_lambda['z'],color='b')
    #plt.axvline(ArLi_like_lambda['a'],color='r')
    #plt.axvline(ArLi_like_lambda['j'],color='r')
    #plt.axvline(ArLi_like_lambda['q'],color='r')
    #plt.axvline(ArLi_like_lambda['r'],color='r')
    #plt.axvline(ArLi_like_lambda['s'],color='r')
    #plt.axvline(ArLi_like_lambda['t'],color='r')
    plt.axvline(Wlambda['px375'],color='g')
    temp = ['r','g','b','k','c','y']
    inp = 0

    temp2 = [['W',43,'c'],
             ['W',44,'c'],
             ['W',45,'c'],
             ['Ar',16,'b']]
    
    for j in semiEmperical:
        plt.axvline(j,color='g')



    temp = scipy.sum(data.data,axis=0)
    idx = temp != 0
    temp -= scipy.median(temp[idx])


    ydata = temp/scipy.sum(temp[idx])
  
    l, = plt.plot(data.wavelength,ydata, lw=2, color='red')
    #l, = plt.plot(scipy.arange(1024), ydata, lw=2, color='red')
    #plt.ylim((1e3,1e6))#scipy.nanmax(data.data)))
    plt.show()

def plotSIF(shot, init=0, offset=0):
    #global ArHe_like_lambda
    #DATA GET
    data = SIFdata.SIFData(shot)
    fig, ax = plt.subplots()
   
    #plt.axvline(ArHe_like_lambda['w'],color='b')
    #plt.axvline(ArHe_like_lambda['x'],color='b')
    #plt.axvline(ArHe_like_lambda['y'],color='b')
    #plt.axvline(ArHe_like_lambda['z'],color='b')
    #plt.axvline(ArLi_like_lambda['a'],color='r')
    #plt.axvline(ArLi_like_lambda['j'],color='r')
    #plt.axvline(ArLi_like_lambda['q'],color='r')
    #plt.axvline(ArLi_like_lambda['r'],color='r')
    #plt.axvline(ArLi_like_lambda['s'],color='r')
    #plt.axvline(ArLi_like_lambda['t'],color='r')
    plt.axvline(Wlambda['px375'],color='g')
    temp = ['r','g','b','k','c','y']
    inp = 0

    temp2 = [['W',43,'c'],
             ['W',44,'c'],
             ['W',45,'c'],
             ['Ar',16,'b']]
    
    for i in temp2:
        print(i,temp[inp])
        for j in SIFdata.lines[i[0]][i[1]]:
            plt.axvline(j,color=i[2])



    temp = scipy.sum(data.data,axis=0)
    idx = temp != 0
    temp -= scipy.median(temp[idx])


    ydata = temp/scipy.sum(temp[idx])
  
    l, = plt.plot(data.wavelength,ydata, lw=2, color='red')
    #l, = plt.plot(scipy.arange(1024), ydata, lw=2, color='red')
    #plt.ylim((1e3,1e6))#scipy.nanmax(data.data)))
    plt.show()

def plotSIF2(shot, init=0, offset=-50, waves=False, method=None, bounds=None):

    #global ArHe_like_lambda
    #DATA GET
    data = SIFdata.SIFData(shot)

    #plt.axvline(ArHe_like_lambda['w'],color='b')
    #plt.axvline(ArHe_like_lambda['x'],color='b')1.7    #plt.axvline(ArHe_like_lambda['y'],color='b')
    #plt.axvline(ArHe_like_lambda['z'],color='b')
    #plt.axvline(ArLi_like_lambda['a'],color='r')
    #plt.axvline(ArLi_like_lambda['j'],color='r')
    #plt.axvline(ArLi_like_lambda['q'],color='r')
    #plt.axvline(ArLi_like_lambda['r'],color='r')
    #plt.axvline(ArLi_like_lambda['s'],color='r')
    #plt.axvline(ArLi_like_lambda['t'],color='r')        
    #plt.axvline(Wlambda['px375'],color='g')
    temp = ['r','g','b','k','c','y']
    inp = 0

    temp2 = [['W',43,'c'],
             ['W',44,'c'],
             ['W',45,'c'],
             ['Ar',16,'b']]
    
    if waves:
        plt.axvline(Wlambda['px375'],color='g')
        for i in temp2:
            print(i,temp[inp])
            for j in SIFdata.lines[i[0]][i[1]]:
                plt.axvline(j,color=i[2])




    temp = scipy.sum(data.data,axis=0)
    temp -= scipy.median(temp)

    #argmax = scipy.argmax(temp[400:440])

    #print(argmax)

    #offset = -85650.6*Wlambda['px375'] - argmax + 34334.5 + 50 -400
    #xdata = (laxis+offset/(-85650.6))
    xdata = data.wavelength
    ydata = temp/scipy.sum(temp)
    
    inp1 = SIFdata.semiEmperical
    #for i in inp1:
    #    plt.axvline(i,alpha=.4,color='k')
    size = len(inp1)
    #inp = scipy.zeros((3*size+1))
    #inp[size:2*size] = inp1
    #inp[2*size:3*size] = 1.2e-4*scipy.ones((len(inp1),))

    #inp2 = inp[:-1].reshape((3,(len(inp)-1)/3))  
    #inp[-1] = -.0005
    #inp[:size] = ydata[scipy.searchsorted(xdata, inp1)] - inp[-1]

    #temp  = (xdata - scipy.dot(scipy.ones((len(xdata),1)),scipy.atleast_2d(inp[1])).T).T/inp[2]

    #inp[-1] = -.0005
    
    #if not bounds == None:
    #    bounds = scipy.zeros((size*3+1,2))
    #    bounds[:size] = [0,1]
    #    bounds[size:2*size] = scipy.add(scipy.atleast_2d(inp[size:2*size]).T,[[-3e-5,3e-5]])
    #    print(scipy.add(scipy.atleast_2d(inp[size:2*size]).T,[[-5e-5,5e-5]]))
    #    bounds[2*size:3*size] = [0,3e-4]
    #    bounds[-1] = [-1,1]

    #print(inp1)
    #print(inp)
    if not bounds == None:
        inp, bounds = SIFdata._assembleInit(xdata, ydata,bounds=bounds)
    else:
        inp = SIFdata._assembleInit(xdata, ydata)

    
    testdata = SIFdata.gaussian(xdata, inp) 
    #plt.plot(xdata,testdata)
    #plt.show()
    #raise AttributeError

    output = scipy.optimize.minimize(SIFdata.compare2,
                                     inp,
                                     args=(xdata,ydata),
                                     method=method,
                                     bounds=bounds)


    print(inp1-output.x[:-1].reshape((3,size))[1])
    print(output.x[:-1].reshape((3,size)))
    #for i in output.x[2::3]:
    #    plt.axvline(i,alpha=.4,color='r')

    #l, = plt.plot(xdata, ydata, lw=2, alpha=.7)
    #plt.plot(xdata,testdata,alpha=.7,linestyle=':')
    #plt.plot(xdata,SIFdata.gaussian(xdata, output.x),alpha=.7,color='r')
    #for i in xrange(len(SIFdata.semiEmperical)):
    #    tempinp = scipy.concatenate(([output.x[0]],output.x[3*i+1:3*i+4]))
    #    plt.plot(xdata,SIFdata.gaussian(xdata, tempinp),alpha=.7,color='c',linestyle='-.')
    #plt.ylim((1e3,1e6))#scipy.nanmax(data.data)))

    #plt.show()


def plotSIF4(shot):    

    #shotfile = dd.shotfile('SIF',shot)
    #data = shotfile('DataSIF')
    
    data = SIFdata.SIFSData(shot)
    #plt.pcolor(data.time,data.wavelength, data.data)#,cmap='viridis')
    plt.imshow(data.data,cmap='viridis_r')

    #plt.xlim([data.time[0],data.time[-1]])
    #plt.ylim([data.wavelength.min(),data.wavelength.max()])
    plt.show()



def fitData(shot, bounds=None, method=None,tol=None):
    #DATA GET
    data = SIFdata.SIFData(shot)
    xdata = data.wavelength
    results = scipy.zeros((len(data.time),3*len(SIFdata.semiEmperical)+1))
    for i in xrange(len(data.time)): #serial
        #print '{}\r'.format(i),
        results[i] = SIFdata._fitData3(i, xdata, data.data, bounds=bounds, method=method, tol=tol)

    #multiprocess
    #pool = mp.Pool(20)
    #results = {}
    #for i in xrange(len(output)):
        #results[i] = pool.apply_async(SIFdata._fitData,(i,xdata,data.data),{'bounds':bounds,'method':method,'tol':tol})

    #pool.join()
    return results


def plotSIF5(shot, init=0, offset=-50):

    #DATA GET
    data = SIFdata.SIFData(shot)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    #plt.axvline(ArHe_like_lambda['w'],color='b')
    #plt.axvline(ArHe_like_lambda['x'],color='b')
    #plt.axvline(ArHe_like_lambda['y'],color='b')
    #plt.axvline(ArHe_like_lambda['z'],color='b')
    #plt.axvline(ArLi_like_lambda['a'],color='r')
    #plt.axvline(ArLi_like_lambda['j'],color='r')
    #plt.axvline(ArLi_like_lambda['q'],color='r')
    #plt.axvline(ArLi_like_lambda['r'],color='r')
    #plt.axvline(ArLi_like_lambda['s'],color='r')
    #plt.axvline(ArLi_like_lambda['t'],color='r')
    plt.axvline(Wlambda['px375'],color='g')
    temp = ['r','g','b','k','c','y']
    inp = 0

    temp2 = [['W',43,'c'],
             ['W',44,'c'],
             ['W',45,'c'],
             ['Ar',16,'b']]
    
    for i in temp2:
        print(i,temp[inp])
        for j in SIFdata.lines[i[0]][i[1]]:
            plt.axvline(j,color=i[2])
    temp = scipy.sum(data.data,axis=0)
    temp -= scipy.median(temp)

    #argmax = scipy.argmax(temp[400:440])

    #print(argmax)

    #offset = -85650.6*Wlambda['px375'] - argmax + 34334.5 + 50 -400
    #xdata = (laxis+offset/(-85650.6))
    xdata = data.wavelength
    ydata = temp/scipy.sum(temp)
    
    inp1 = SIFdata.semiEmperical
    #for i in inp1:
    #    plt.axvline(i,alpha=.4,color='k')
    size = len(inp1)

    if not bounds == None:
        inp, bounds = SIFdata._assembleInit(xdata, ydata, bounds=bounds)
    else:
        inp = SIFdata._assembleInit(xdata, ydata)

    
    testdata = SIFdata.gaussian(xdata, inp) 
    #plt.plot(xdata,testdata)
    #plt.show()
    #raise AttributeError

    output = scipy.optimize.minimize(SIFdata.compare,
                                     inp,
                                     args=(xdata,ydata),
                                     method=method,
                                     bounds=bounds)



    #l2, = plt.semilogy(laxis2+offset/(-85650.6), data.data[init], lw=2, color='green')
    l, = plt.plot(data.wavelength, sig.medfilt(data.data[init],5), lw=2, color='red')
    plt.ylim((0,5e4))#scipy.nanmax(data.data)))


    axcolor = 'lightgoldenrodyellow'
    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])

    sfreq = DiscreteSlider(axfreq, 'Freq', 0, len(data.time), valinit=init, valfmt='%0.0i', increment=1.0)

    def update(val):
        freq = int(sfreq.val)
        print(freq)
        #l2.set_ydata(data.data[val])
        l.set_ydata(sig.medfilt(data.data[val],5))
        fig.canvas.draw_idle()
    sfreq.on_changed(update)
    plt.show()

def plotSIF6(shot, idx, bounds=None, method=None, tol=None):

    data = SIFdata.SIFData(shot)
    xdata = data.wavelength
    results = SIFdata._fitData(idx, xdata, data.data, bounds=bounds, method=method, tol=tol)
    #print(results)
    plotout(xdata,data.data[idx],results)
    plt.show()
    return results

def plotSIF7(shot, idx, bounds=None, method=None, tol=None):

    data = SIFdata.SIFData(shot)
    xdata = data.wavelength
    results = SIFdata._fitData3(idx, xdata, data.data, bounds=bounds, method=method, tol=tol)
    #print(results)
    plotout(xdata,data.data[idx],results)
    plt.show()
    return results


def plotout(xdata, ydata, inp):
    #show the input vs output of the fit versus the data given by xdata, ydata and the inp
    temp = SIFdata._assembleInit(xdata, ydata)
    plt.plot(xdata, ydata)
    plt.plot(xdata, _gauss.gauss(xdata,inp[1:])+inp[0],'c')
    for i in xrange((len(inp)-1)/3):
        plt.plot(xdata,inp[0]+_gauss.gauss(xdata, inp[3*i+1:3*(i+1)+1]),'r:')
    for i in xrange((len(temp)-1)/3):
        plt.plot(xdata,temp[0]+_gauss.gauss(xdata, temp[3*i+1:3*(i+1)+1]),'b-.',alpha=.7)



def plotSIF8(shot, init=0, offset=-50):

    #DATA GET
    data = SIFdata.SIFData(shot)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    #plt.axvline(ArHe_like_lambda['w'],color='b')
    #plt.axvline(ArHe_like_lambda['x'],color='b')
    #plt.axvline(ArHe_like_lambda['y'],color='b')
    #plt.axvline(ArHe_like_lambda['z'],color='b')
    #plt.axvline(ArLi_like_lambda['a'],color='r')
    #plt.axvline(ArLi_like_lambda['j'],color='r')
    #plt.axvline(ArLi_like_lambda['q'],color='r')
    #plt.axvline(ArLi_like_lambda['r'],color='r')
    #plt.axvline(ArLi_like_lambda['s'],color='r')
    #plt.axvline(ArLi_like_lambda['t'],color='r')
    plt.axvline(Wlambda['px375'],color='g')
    temp = ['r','g','b','k','c','y']
    inp = 0

    temp2 = [['W',43,'c'],
             ['W',44,'c'],
             ['W',45,'c'],
             ['Ar',16,'b']]
    
    for i in temp2:
        print(i,temp[inp])
        for j in SIFdata.lines[i[0]][i[1]]:
            plt.axvline(j,color=i[2])



    l, = plt.plot(data.wavelength, data.data[init], lw=2, color='green')
    l2, = plt.plot(data.wavelength, sig.medfilt(data.data[init],5), lw=2, color='red')
    temp = SIFdata._assembleInit(data.wavelength, data.data[init])
    l3, = plt.plot(data.wavelength, _gauss.gauss(data.wavelength,temp[1:])+temp[0], lw=2.,color='c')
    l4, = plt.plot(data.wavelength, SIFdata._fitData4(init, data.wavelength, data.data, bounds=True), lw=2.,color='m',alpha=.5)

    plt.ylim((0,5e4))#scipy.nanmax(data.data)))


    axcolor = 'lightgoldenrodyellow'
    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])

    sfreq = DiscreteSlider(axfreq, 'Freq', 0, len(data.time), valinit=init, valfmt='%0.0i', increment=1.0)

    def update(val):
        freq = int(sfreq.val)
        print(freq)
        l.set_ydata(data.data[val])
        l2.set_ydata(sig.medfilt(data.data[val],5))
        temp2 = sig.medfilt(data.data[val],5)
        temp = SIFdata._assembleInit(data.wavelength, data.data[val])
        l3.set_ydata(_gauss.gauss(data.wavelength,temp[1:])+temp[0])
        l4.set_ydata(SIFdata._fitData4(val, data.wavelength, data.data, bounds=True,tol=1e-6))
        fig.canvas.draw_idle()
    sfreq.on_changed(update)
    plt.show()


def plotSIF9(shot, init=0, offset=0):

    #DATA GET
    data = SIFdata.SIFData(shot,offset=offset)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    #plt.axvline(ArHe_like_lambda['w'],color='b')
    #plt.axvline(ArHe_like_lambda['x'],color='b')
    #plt.axvline(ArHe_like_lambda['y'],color='b')
    #plt.axvline(ArHe_like_lambda['z'],color='b')
    #plt.axvline(ArLi_like_lambda['a'],color='r')
    #plt.axvline(ArLi_like_lambda['j'],color='r')
    #plt.axvline(ArLi_like_lambda['q'],color='r')
    #plt.axvline(ArLi_like_lambda['r'],color='r')
    #plt.axvline(ArLi_like_lambda['s'],color='r')
    #plt.axvline(ArLi_like_lambda['t'],color='r')
    plt.axvline(Wlambda['px375'],color='g')
    temp = ['r','g','b','k','c','y']
    inp = 0

    temp2 = [['W',43,'c'],
             ['W',44,'c'],
             ['W',45,'c'],
             ['Ar',16,'b']]
    
    for i in temp2:
        print(i,temp[inp])
        for j in SIFdata.lines[i[0]][i[1]]:
            plt.axvline(j,color=i[2])



    l, = plt.semilogy(data.wavelength, data.data[init], lw=2, color='green')
    l2, = plt.semilogy(data.wavelength, sig.medfilt(data.data[init],5), lw=2, color='red')
    temp = SIFdata._assembleInit(data.wavelength, data.data[init])
    l3, = plt.semilogy(data.wavelength, _gauss.gauss(data.wavelength,temp[1:])+temp[0], lw=2.,color='c')
    l4, = plt.semilogy(data.wavelength, SIFdata._fitData4(init, data.wavelength, data.data, bounds=True), lw=2.,color='m',alpha=.5)
    #l5, = plt.semilogy(data.wavelength, SIFdata._fitData9(init, data.wavelength, data.data, bounds=True), lw=2.,color='k',alpha=.5)
    l6, = plt.semilogy(data.wavelength, SIFdata._fitData8(init, data.wavelength, data.data, bounds=True), lw=2.,color='b',alpha=.5)

    plt.ylim((1e3,5e4))#scipy.nanmax(data.data)))


    axcolor = 'lightgoldenrodyellow'
    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])

    sfreq = DiscreteSlider(axfreq, 'Freq', 0, len(data.time), valinit=init, valfmt='%0.0i', increment=1.0)

    def update(val):
        freq = int(sfreq.val)
        print(freq)
        l.set_ydata(data.data[val])
        l2.set_ydata(sig.medfilt(data.data[val],5))
        temp2 = sig.medfilt(data.data[val],7)
        temp = SIFdata._assembleInit(data.wavelength, data.data[val])
        l3.set_ydata(_gauss.gauss(data.wavelength,temp[1:])+temp[0])
        l4.set_ydata(SIFdata._fitData4(val, data.wavelength, data.data, bounds=True,tol=1e-8))
        #l5.set_ydata(SIFdata._fitData9(val, data.wavelength, data.data, bounds=True,tol=1e-8))
        l6.set_ydata(SIFdata._fitData8(val, data.wavelength, data.data, bounds=True,tol=1e-8))

        fig.canvas.draw_idle()
    sfreq.on_changed(update)
    plt.show()
