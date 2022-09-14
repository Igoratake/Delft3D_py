
import sys
import numpy as np
import pandas as pd
import xarray as xr
import os
from datetime import datetime
from .GridTools import *

def GetSrcPoint(grdname, x, y):

    g = ReadD3DGrid(grdname)
    dist = np.sqrt((g['x']-x)**2+(g['y']-y)**2)
    m, n = np.unravel_index(
                            np.argmin(dist, axis=None),
                            dims=g['x'].shape
                            )
    return m, n


def ReadSrc(fname):

    src = pd.read_table(
        fname,
        sep='\s+',
        header=None,
        index_col=0
    )
    try:
        src.columns = ['Interpolation', 'M', 'N', 'K', 'Type']
    except:
        src.columns = ['Interpolation', 'M', 'N', 'Type']
    return src


def WriteSrc(src, fname):

    with open(fname,'w+') as f:
        for index, val in src.iterrows():
            f.write(index)
            spaces = 22-len(index)
            f.write(' ' * spaces)
            for v in val:
                f.write(str(v)+' ')

def ReadDis(fname):

    # gets Name
    f = open(fname, 'r')
    lines = f.readlines()
    nloc = len([line for line in lines if 'location' in line])
    data = list()
    dis = pd.DataFrame(
        columns=[
            'location',
            'contents',
            'time-function',
            'reference-time',
            'interpolation',
            'table-name',
            'time-unit',
            'records-in-table',
            'parameter',
        ],
        index=range(nloc)
    )

    f.close()
    i = 0
    data = list()
    pars = ''
    for j, line in enumerate(lines):

        col = line.split()[0]
        if col in dis.columns:
            if "'" in line:
                info = "".join(
                    line.split("'")[1].split('(')[0].split(':')[-1].split()[0]
                )

                if 'parameter' in line:
                    #if info not in pars:                        
                    pars = pars + info+'__'
                else:
                    dis.loc[i, col] = info
            else:
                dis.loc[i, col] = line.split()[-1]               

        if 'records-in-table' in line:
           
            dis.loc[i, 'parameter'] = pars.split('__')[:-1]
            data.append(
                np.genfromtxt(
                    fname,
                    skip_header=j + 1,
                    max_rows=int(dis.loc[i, col])
                )
            )
            i += 1

    dis.set_index(['location'], drop=True, inplace=True)
    dis = dis.T.to_dict()    

    for k in dis.keys():

        i = int(dis[k]['table-name'].split(':')[-1])
        
        for j, parameter in enumerate(dis[k]['parameter']):
            
            dis[k][parameter] = data[i - 1][:, j]
        reftime = datetime.datetime.strptime(str(dis[k]['reference-time']), '%Y%m%d')
        dis[k]['datetime'] = pd.DatetimeIndex(
            data[i - 1][:, 0].astype('timedelta64[m]') + np.datetime64(reftime)
        )
    return dis


def WriteDis(mdict, fname):

    f = open(fname, 'w+')
    f.close()
    for k, params in mdict.items():        
        
        f = open(fname, 'a')
        f.write(
            "table-name\t'Discharge: {}'\n".format(
                mdict[k]['table-name']
            )
        )
        f.write("contents\t'{}'\n".format(mdict[k]['contents']))
        f.write("location\t'{}'\n".format(k))
        f.write("time-function\t'{}'\n".format(mdict[k]['time-function']))
        f.write("reference-time\t {}\n".format(mdict[k]['reference-time']))
        f.write("time-unit\t'{}'\n".format(mdict[k]['time-unit']))
        f.write("interpolation\t'{}'\n".format(mdict[k]['interpolation']))
        # concatenate data parameters
        for param in mdict[k]['parameter']:
            if param == 'Salinity':
                ndata = mdict[k][param]
                units = 'ppt'
            elif param == 'Temperature':
                ndata = mdict[k][param]
                units = u'°C'
            elif param == 'time':
                units = 'min'
                data = mdict[k][param]                
            elif param == 'flux/discharge':
                ndata = mdict[k][param]
                units = 'm3/s'
                param = param+' rate'
            else:
                ndata = mdict[k][param]
                units = 'kg/m3'
            f.write("parameter\t'{}' unit '[{}]'\n".format(param, units))
            if 'ndata' in locals():
                if len(data) == len(ndata):                    
                    data = np.c_[data, ndata]                

        f.write("records-in-table\t{}\n".format(mdict[k]['records-in-table']))
        f.close()
        f = open(fname, 'ab')
        del (ndata)
        np.savetxt(
            f, data, fmt='%0.8f')
        f.close()
