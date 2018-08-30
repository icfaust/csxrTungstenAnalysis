import _gauss
import scipy
import scipy.optimize
import dd
import sqlite3 # I could use sqlalchemy, but this initial table is probably easiest with string calls
import multiprocessing as mp

#                            ~W44+         ~W45+         W-line       ~W43+ ~W?+  X-line Y-line ~W44+ ~W44+  Z-line
semiEmperical = scipy.array([.3909,.39316,.39331,.39374,.39492,.3955,.3963,.39648,.3966,.39695,.3975,.39887,.39943])







#################################################################################################################################                                                                               #
#                                                                                                                               #
#                                                                                                                               #
#                                                                                                                               #
#                                                                                                                               #
#                                                                                                                               #
#                                                                                                                               #
#                                                                                                                               #
#                           This is for analysis using maximum likelihood, and is a copy of repair.py                           #
#                                                                                                                               #
#                                                                                                                               #
#                                                                                                                               #
#                                                                                                                               #
#                                                                                                                               #
#                                                                                                                               #
#                                                                                                                               #
#################################################################################################################################







#2017 calibration off of 34228 w line
calib = 33636.23 #33354.6# 33636.23
disp = 86363.6363 #85650.6# 86363.6363

_tablename = 'SIFdataMLE2015.db'
_olddb = 'SIFdata5.db'

# internal fitting bounds and initial value for baseline and gaussian height, offset, and width
_bounds = scipy.array([[1,65536],[0,scipy.inf],[-3e-5,3e-5],[2e-5,3e-4]])

_initial = scipy.array([0.,1.,1.,.7e-4,0.])

def interface(inp, xdata, ydata):
    """ splits the c function output to two variables, the RMSE and the derivative of RMSE with respect to the parameters"""

    p = _gauss.gpmle(xdata,ydata,inp)
    return p[0],p[1:]


def _assembleInit(xdata, ydata, bounds=None):
    """assemble the initial values based on the spectrum"""

    #INTITIALIZE VARIABLES
    init = scipy.zeros((3*len(semiEmperical)+1,))
    output = scipy.zeros((len(init), 2))
    ones = scipy.ones(semiEmperical.shape)
    
    #set baseline
    axis = scipy.mgrid[-32:65569:64]
    init[0] = (scipy.histogram(ydata,bins=axis)[0])[:pow(2,9)].argmax()*64 #ydata.min()#_initial[0] This makes the initial baseline
    #limited to the lower part of the measured ADC, which will prevent saturation events from becoming the 'baseline'
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


def SIFData(shot, filename="SIF", data="DataSIF", offset=0, rev=True):
    """This yanks data for the analysis (for fitting gaussians)"""
    shotfile = dd.shotfile(filename,shot)
    temp = shotfile(data)
    temp.data = temp.data[:,:1024]
    temp.data = temp.data[:,::-1] #flip axis for wavelength data
    temp.wavelength = (scipy.arange(temp.data.shape[1]) + offset + calib)/disp
    temp.wavelength = temp.wavelength[:1024]

    if len(temp.data)*1.5 < len(temp.time): #this corrects an error where the timebase of the SIF data is incorrect (x2 too many points)
        temp.time = temp.time[::2]
    #timebase issue correction
    #if len(temp.data) != len(temp.time):
    #    raise dd.PyddError #timebase problem

    #temp.wavelength = (scipy.arange(1250) - 34334.5 + offset)/(-85650.6) #give it a wavelength parameter.  This is hardcoded and derived from Sertoli's work in /u/csxr/sif/sif.pro
    return temp #send it out into the wild


def GIWData(shot, filename="GIW", data="c_W"):
    shotfile = dd.shotfile(filename,shot)
    return shotfile(data)


def goodGIW(time, shot, name="c_W"):
    """extract values which are only valid, otherwise place in zeros"""
    temp = GIWData(shot, data=name)
    interp = scipy.interpolate.interp1d(temp.time,
                                        temp.data,
                                        bounds_error=False)
    output = interp(time)
    output[ scipy.logical_not(scipy.isfinite(output))] = 0. #force all bad values to zero
    return output   


def fitShot(shot, bounds=True, method=None, tol=None, serial=True, offset=0):
    """interface for fitting all the time points of the spectrometer"""
    data = SIFData(shot,offset=offset)
    return fitData(data, bounds=bounds, method=method, tol=tol, serial=serial)


def fitData(data, bounds=True, method=None, tol=None, serial=True):
    """interface for fitting all the time points of the spectrometer"""
    xdata = data.wavelength

    if serial:
        #serial
        results = scipy.zeros((len(data.data),3*len(semiEmperical)+2))
        for i in xrange(len(data.time)): #serial
            #print '{}'.format(i),
            results[i] = _fitData(i, xdata, data.data[i]-1000, bounds=bounds, method=method, tol=tol) #1000 if for ADC offset
    else:

        #multiprocess
        pool = mp.Pool(35)
        output = {}
        for i in xrange(len(data.data)):
            output[i] = pool.apply_async(_fitData,(i,xdata,data.data[i]- 1000),{'bounds':bounds,'method':method,'tol':tol})#1000 if for ADC offset
        pool.close()
        pool.join()
        results = scipy.array([output[i].get() for i in output])
    
    return results


def _fitData(idx, xdata, ydata, bounds=None, method=None, tol=None):
    """ internal program to interface with scipy.optimize.minimize (BFGS_L) algo """

    #ydata =data[idx]

    # need to subtract baseline from data


    if not bounds == None:
        inp, bounds = _assembleInit(xdata, ydata, bounds=bounds)
        bounds[1::3] *= bounds[3::3]*scipy.sqrt(scipy.pi)/123. #same for these values 
    else:
        inp = _assembleInit(xdata, ydata)
    #inp[0] *= .9     

    inp[0] /= 123.
    inp[1::3] *= inp[3::3]*scipy.sqrt(scipy.pi)/123. #convert to integrated counts
    # the 123 comes from converting counts to actual photons for the energies observed by the SIF
        
    temp = scipy.optimize.minimize(interface,
                                   inp,
                                   args=(xdata,ydata/123.),
                                   method=method,
                                   bounds=bounds,
                                   jac=True)
   
    output = temp.x
    if scipy.isfinite(temp.fun):
        output = scipy.append(output,[temp.fun])
    else:
        output = scipy.append(output,[-1.])


    return output

def writeData(shot, results, timebase, conn, name, idx):
    """ this function writes to the SQL database the new data derived from fitData
    the format is as follow:

    RMSE of fit, shotnumber, time, params (with peak val, offset, width)"""
    c = conn.cursor()
    text = 'INSERT INTO '+name+' VALUES('

    for i in xrange(len(results)):
        newstr = text+str(idx)+', '+str(int(shot))+', '+str(timebase[i])+', '+str(results[i][-1])
        for j in results[i][:-1]:
            newstr +=', '+str(j)
            
        newstr += ',0,0)'
        #print(newstr)
        c.execute(newstr)
        idx = idx+1

    conn.commit() # write data to database
    return idx

def reWriteData(shot, results, timebase, conn, name, idx, c_w_qc, c_w_l):
    """ this function writes to the SQL database the new data derived from fitData
    the format is as follow:

    RMSE of fit, shotnumber, time, params (with peak val, offset, width)"""
    c = conn.cursor()
    text = 'INSERT INTO '+name+' VALUES('

    for i in xrange(len(results)):
        newstr = text+str(idx[i])+', '+str(int(shot))+', '+str(timebase[i])+', '+str(results[i][-1])
        for j in results[i][:-1]:
            newstr +=', '+str(j)
            
        newstr += ', '+str(c_w_qc[i])+', '+str(c_w_l[i])+ ')'
        #print(newstr)
        c.execute(newstr)

    conn.commit() # write data to database


def findTime(shot):
    shotfile = dd.shotfile('EQH',shot)
    time = shotfile('time').data
    del(shotfile)
    return [time[0],time[-1]]
    #these are the ranges in which to store SIF data


def analyzeShot(shot, conn, name, idx, bounds=True, method=None, tol=None, serial=True):
    try:
        #find data
        lims = findTime(shot)
        data = SIFData(shot)
        if lims[1] > data.time[-1]:
            raise dd.PyddError #timebase is way wrong, toss it

        #solve for indicies of data to be fitted
        indices = scipy.searchsorted(data.time,lims)

        #snip snip
        data.data = data.data[indices[0]:indices[1]]
        data.time = data.time[indices[0]:indices[1]]

        #fit the data
        output = fitData(data, bounds=bounds, method=method, tol=tol, serial=serial)
        
        #write to sql database
        idx = writeData(shot, output, data.time, conn, name, idx)
        
        return idx
    except dd.PyddError:
        #if there is any error pulling the data (SIF or EQH), toss the shot
        return idx



def reAnalyzeShot(shot, conn, name, idx, time, c_w_qc, c_w_l, bounds=True, method=None, tol=None, serial=True):
    try:
        #find data


        data = SIFData(shot)

        #solve for indicies of data to be fitted
        indices = scipy.searchsorted(data.time, time-1e-5) #the -1e-5 solves an issue with the data.time being a double, and time being a float

        #snip snip
        data.data = data.data[indices]
        data.time = data.time[indices]

        #fit the data
        output = fitData(data, bounds=bounds, method=method, tol=tol, serial=serial)
        
        #write to sql database
        idx = reWriteData(shot, output, data.time, conn, name, idx, c_w_qc, c_w_l)
        
        return idx
    except dd.PyddError:
        #if there is any error pulling the data (SIF or EQH), toss the shot
        return idx

def reAnalyzeShot2(shot, conn, idx, time, bounds=True, method=None, tol=None, serial=True, offset=9):
    try:
        #find data
        curs= conn.cursor()

        data = SIFData(shot, offset=offset)

        #solve for indicies of data to be fitted
        indices = scipy.searchsorted(data.time, time-1e-5) #the -1e-5 solves an issue with the data.time being a double, and time being a float

        #snip snip
        data.data = data.data[indices]
        data.time = data.time[indices]

        #fit the data
        output = fitData(data, bounds=bounds, method=method, tol=tol, serial=serial)
        print(output.shape)
        
        #modify sql database
        print(idx.shape)
        for i in xrange(len(idx)):
            changeValue(curs, idx[i], output[i])
        
        return idx
    except dd.PyddError:
        #if there is any error pulling the data (SIF or EQH), toss the shot
        return idx

def run(shots=[32812,34995], idx=0, name='shots2016'):
    conn = sqlite3.connect(_tablename)
    for i in scipy.arange(shots[0],shots[1]+1):
        print(idx)
        idx = analyzeShot(i, conn, name, idx, bounds=True, serial=False)
    conn.close()
    return idx

def rerun(name='shots2016',startidx=0):
    conn = sqlite3.connect(_tablename)

    oldConn = sqlite3.connect(_olddb)
    oldC = oldConn.cursor()

    # extract necessary idx, time, shot c_w_qc and c_w_l
    idx = scipy.squeeze((oldC.execute('SELECT id from '+name)).fetchall())
    
    newstart = scipy.searchsorted(idx,[startidx])[0]
    time = scipy.squeeze((oldC.execute('SELECT time from '+name)).fetchall())[newstart:]
    c_w_qc = scipy.squeeze((oldC.execute('SELECT c_W from '+name)).fetchall())[newstart:]
    c_w_l = scipy.squeeze((oldC.execute('SELECT c_w_l from '+name)).fetchall())[newstart:]
    shots = scipy.squeeze((oldC.execute('SELECT shot from '+name)).fetchall())[newstart:]
    idx = idx[newstart:]

    uniq = scipy.unique(shots)
    for i in uniq:
        idxin = shots == i
        # find index which matches current shot

        reAnalyzeShot(i, conn, name, idx[idxin], time[idxin], c_w_qc[idxin], c_w_l[idxin], bounds=True, serial=False)


def modify(name='shots2016'):

    shots = scipy.array([34401,
                         34664,
                         34665,
                         34666,
                         34667,
                         34668,
                         34669,
                         34670,
                         34671,
                         34672,
                         34673,
                         34674])

    conn = sqlite3.connect(_tablename)
    oldC = conn.cursor()

    for i in shots:
    # extract necessary idx, time, shot c_w_qc and c_w_l
        idx = scipy.squeeze((oldC.execute('SELECT id from '+name+' where shot='+str(i))).fetchall())
        time = scipy.squeeze((oldC.execute('SELECT time from '+name+' where shot='+str(i))).fetchall())
        print(time.shape,idx.shape)

        reAnalyzeShot2(i, conn, idx, time, bounds=True, serial=False)

    conn.commit()


def genTable(name='shots2016'):

    conn = sqlite3.connect(_tablename)
    c = conn.cursor()
    temp = 'CREATE TABLE '+name+' (id INTEGER PRIMARY KEY ASC, shot INTEGER, time REAL, '
    temp += 'chi REAL, baseline REAL'
    for i in xrange(len(semiEmperical)):
        temp += ', val'+str(i)+' REAL,'
        temp += ' offset'+str(i)+' REAL,'
        temp += ' width'+str(i)+' REAL'

    temp += ', c_W REAL, c_W_l REAL)'
    print(temp)
    c.execute(temp)
    conn.commit()
    conn.close()


def addColumn(name='c_w_qc', table='shots2016', datatype='FLOAT'):
    conn = sqlite3.connect(_tablename)
    c = conn.cursor()
    temp = 'ALTER TABLE '+table+' ADD COLUMN '+name+' '+datatype
    c.execute(temp)
    conn.commit()
    conn.close()


def addGIWdata(conn, shot, name='c_w_qc', name2='c_W', table='shots2016'):
    """pulls all time points associated with shot and places in database"""
    c = conn.cursor()
    points = scipy.squeeze((c.execute('SELECT id from '+table+' where shot='+str(shot))).fetchall())
    time = scipy.squeeze((c.execute('SELECT time from '+table+' where shot='+str(shot))).fetchall())
    
    #if the shotfile doesn't exist, or if good data at specific timepoints dont exist, set to 0
    try:
        output = goodGIW(time, shot, name2)
    except dd.PyddError:    
        print(shot)
        output = scipy.zeros(time.shape)

    for i in xrange(len(points)):
        c.execute('UPDATE '+table+' SET "'+name+'"='+str(output[i])+' WHERE id='+str(points[i]))

    conn.commit()
    #print(shot),


def findShots(conn, table='shots2016'):
    c = conn.cursor()
    c.execute('SELECT shot from '+table)
    temp = scipy.squeeze(c.fetchall())
    output = scipy.unique(temp)
    return output


def updateGIW(name='c_w_qc',name2='c_W',table='shots2016'):
    """ add all GIW data to table """
    connection = sqlite3.connect(_tablename)
    
    shots = findShots(connection, table=table)
    for i in shots:
        addGIWdata(connection, i, name=name, name2=name2, table=table)

    connection.close()


def badGIW(table='shots2016'):
    """ turns out some valid shots have no GIW data (even though data and Thomson data exists), list them so that I can check them independently"""

    conn = sqlite3.connect(_tablename)
    shots = findShots(conn, table=table)
    conn.close()

    badshots = []
    idx = []
    outidx = 0
    for i in shots:
        try:
            temp = GIWData(i)
        except dd.PyddError:
            idx += [outidx]
            badshots += [i]
            outidx += 1

    for i,j in zip(idx,badshots):
        print(i,j)

    return badshots,idx


def changeValue(cursor, idx, values):
    names = ['baseline','val0','offset0','width0',
             'val1','offset1','width1',
             'val2','offset2','width2',
             'val3','offset3','width3',
             'val4','offset4','width4',
             'val5','offset5','width5',
             'val6','offset6','width6',
             'val7','offset7','width7',
             'val8','offset8','width8',
             'val9','offset9','width9',
             'val10','offset10','width10',
             'val11','offset11','width11',
             'val12','offset12','width12']
    
    text = 'UPDATE shots2016 SET '
    for i in xrange(len(names)):
        newtext = text+names[i]+'='+str(values[i])+' WHERE id='+str(idx)
        print(newtext)
        cursor.execute(newtext)

def changeData(shotlist, data, idx, name='shots2016'):

    tedata = loadTe()

    conn = sqlite3.connect(_dbname)
    
    unishots = scipy.unique(shotlist)

    for i in unishots:
        inp = shotlist == i
        gen(conn, tedata, data[inp], idx[inp], timelist[inp], int(i), name=name)


    conn.close()
