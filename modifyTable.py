# this program is to correct an issue in the table I generated using SIFweightings
# There needs to be an additional multiple of ne multiplied through (so that it is ne
# squared) This will likely make things more core-weighted in the kernel, which should
# be to its benefit anyway.  Here is to hoping it isn't too singular

import scipy
import scipy.io
import GIWprofiles
import sqlite3
import dd

#GIW data load

_dbname = 'SIFweights.db'

_olddbname = 'SIFweight.db'
_olddbname2 = 'SIFdata2.db'


def goodGIW(time, shot, name="c_W"):
    """extract values which are only valid, otherwise place in zeros"""
    temp = GIWData(shot, data=name)
    interp = scipy.interpolate.interp1d(temp.time,
                                        temp.data,
                                        bounds_error=False)
    output = interp(time)
    output[ scipy.logical_not(scipy.isfinite(output))] = 0. #force all bad values to zero
    return output


def GIWData(shot, filename="GIW", data="c_W"):
    shotfile = dd.shotfile(filename,shot)
    return shotfile(data)

def gen(conn, tedata, data, idx, time, shot, name='shots2016'):

    # GET THAT SIIIIICK DATA YEAH
    tped = goodGIW(time, shot, name="t_e_ped")
    tcore = goodGIW(time, shot, name="t_e_core")

    nped = goodGIW(time, shot, name="n_e_ped")
    ncore = goodGIW(time, shot, name="n_e_core")

    output = scipy.zeros(data.shape)

    for i in xrange(len(time)):
        print(i),
        ne = GIWprofiles.ne(GIWprofiles.te2rho2(tedata, tcore[i], tped[i]), ncore[i], nped[i])
        ne[scipy.logical_not(scipy.isfinite(ne))] = 0.

        output[i] = data[i]*ne
        #multiply for new data

    

    writeData(idx, output, conn, name)
        #place 


def modify(shotlist, timelist, data, idx, name='shots2016'):

    tedata = loadTe()

    conn = sqlite3.connect(_dbname)
    
    unishots = scipy.unique(shotlist)

    for i in unishots:
        inp = shotlist == i
        gen(conn, tedata, data[inp], idx[inp], timelist[inp], int(i), name=name)


    conn.close()

def yank(name='shots2016'):
    conn1 = sqlite3.connect(_olddbname)
    c = conn1.cursor()
    temp = scipy.squeeze((c.execute('SELECT * FROM '+name)).fetchall())
    data = temp[:,1:]
    idx = temp[:,0].astype(int)
    print('large dataset yanked')
    
    conn2 = sqlite3.connect(_olddbname2)
    c = conn2.cursor()
    shot = scipy.squeeze((c.execute('SELECT shot FROM '+name)).fetchall()).astype(int)
    time = scipy.squeeze((c.execute('SELECT time FROM '+name)).fetchall())
    idx2 = scipy.squeeze((c.execute('SELECT id FROM '+name)).fetchall()).astype(int)
    print('shot and time params yanked')
    shotlist = scipy.zeros(idx.shape).astype(int)
    timelist = scipy.zeros(idx.shape)
    for i in xrange(len(idx)):
        idxtemp = idx2 == idx[i]
        shotlist[i] = shot[idxtemp][0]
        timelist[i] = time[idxtemp][0]

    conn1.close()
    conn2.close()
    

    return shotlist, timelist, data, idx


def genTable(name='shots2016'):

    te = loadTe()
    #print(len(temp))

    conn = sqlite3.connect(_dbname)
    c = conn.cursor()
    temp = 'CREATE TABLE '+name+' (id INTEGER PRIMARY KEY ASC '
    for i in xrange(len(te)):
        temp += ', val'+str(i)+' REAL'

    temp += ')'

    #print(temp)
    c.execute(temp)
    conn.commit()
    conn.close()


def writeData(idx, results, conn, name):
    """ this function writes to the SQL database the new data derived from fitData
    the format is as follow:

    RMSE of fit, shotnumber, time, params (with peak val, offset, width)"""
    c = conn.cursor()
    text = 'INSERT INTO '+name+' VALUES('

    for i in xrange(len(results)):
        newstr = text+str(idx[i])
        for j in results[i]:
            newstr +=', '+str(j)
            
        newstr += ')'
        #print(newstr)
        c.execute(newstr)

    conn.commit() # write data to database


def loadTe(name='W_Abundances_grid_puestu_adpak_fitscaling_74_0.00000_5.00000_1000_idlsave'):
    data = scipy.io.readsav(name)
    return data['en']

