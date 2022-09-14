
import sys
import numpy as np
import pandas as pd
import xarray as xr
import os
from datetime import datetime
from h5py import File
from TTutils.Utils import matlabtime2pythontime


def ReadBnd(fname):

    bnd = pd.read_csv(
        fname,
        sep='\s+',
        header=None,
        index_col=0
    )

    if bnd.shape[1] == 8:
        bnd.columns = [
            'Type',
            'Force',
            'M1',
            'N1',
            'M2',
            'N2',
            'Alpha',
            'VerticalProfile'
        ]
    elif bnd.shape[1] == 7:
        bnd.columns = ['Type', 'Force', 'M1', 'N1', 'M2', 'N2', 'Alpha']
    elif bnd.shape[1] == 9:
        bnd.columns = [
            'Type',
            'Force',
            'M1',
            'N1',
            'M2',
            'N2',
            'Alpha',
            'VerticalProfile1',
            'VerticalProfile2'
        ]
    bnd.index.name = 'Name'
    return bnd

def WriteBnd(bnd, fname):

    fname, _ = os.path.splitext(fname)
    fname = fname + '.bnd'
    columns = [
        'Type',
        'Force',
        'M1',
        'N1',
        'M2',
        'N2',
        'Alpha',
        'VerticalProfile'
    ]
    if isinstance(bnd, pd.DataFrame):
        bnd.loc[:, columns].to_table(fname, header=None, sep='\t')
    elif isinstance(bnd, np.array):
        bnd = pd.DataFrame(bnd[:, 1:], index=bnd[:, 0])
        bnd.columns = columns
        bnd.loc[:, columns].to_table(fname, header=None, index=None, sep='\t')
    if isinstance(bnd.bnd, pd.DataFrame):
        bnd.loc[:, columns].to_table(fname, header=None, sep='\t')

def CreateBCA(mdf, data, TPXO=True):

    if TPXO:
        ds = xr.open_dataset('http://opendap.deltares.nl/thredds/dodsC/\
        opendap/deltares/delftdashboard/tidemodels/tpxo72.nc')

    x = mdf.grd['x']
    y = mdf.grd['y']

    # for index, bnd in mdf.bnd.iterrows():
    pass


def ConvertSSHMat2NC(matfile):

    mat = File(matfile)
    S1 = mat['data']['Val'].value
    XCOR = mat['data']['X'].value
    YCOR = mat['data']['Y'].value
    XCOR = XCOR[:-1, :-1]+XCOR[1:, 1:]+XCOR[:-1, 1:]+XCOR[1:, :-1]
    YCOR = YCOR[:-1, :-1]+YCOR[1:, 1:]+YCOR[:-1, 1:]+YCOR[1:, :-1]
    YCOR = YCOR[1:-1,1:-1]
    S1 = S1[1:-1,1:-1,:]
    XCOR = XCOR[1:-1,1:-1]

    time = mat['data']['Time'].value.squeeze()
    time = [matlabtime2pythontime(t) for t in time]

    S1 = np.transpose(S1, (2, 0, 1))

    ds = xr.Dataset({
        'S1': (['time', 'x', 'y'], S1)
    },
    coords={
                'XCOR': (['x', 'y'], XCOR),
                'YCOR': (['x', 'y'], YCOR),
                'time':  (['time'], time)
            }
    )
    return ds

def NestingREMObct(mdf, fname, BndCoordConvert=None, NestBoundaries=None):

    if isinstance(fname, str):
        ds = xr.open_dataset(fname)
    elif isinstance(fname, xr.core.dataset.Dataset):
        ds = fname
    else:
        raise ValueError('TrimName must be a trimfilename')

    if NestBoundaries:
        if isinstance(NestBoundaries, (list, str)):
            mdf.bnd = mdf.bnd.loc[NestBoundaries, :]

    mdf.bnd.loc[:, ['M1', 'N1', 'M2', 'N2']] -= 1

    if isinstance(BndCoordConvert, (list, np.ndarray)):
        ini, end = BndCoordConvert
        mdf.grd['x'], mdf.grd['y'] = CoonvertCoordinates(
            mdf.grd['x'],
            mdf.grd['y'],
            ini,
            end
        )

    Nmax, Mmax = mdf.grd['x'].shape
    Mmax -= 1
    Nmax -= 1

    mdf.bnd.loc[mdf.bnd.M1 > Mmax, 'M1'] = Mmax
    mdf.bnd.loc[mdf.bnd.M2 > Mmax, 'M2'] = Mmax
    mdf.bnd.loc[mdf.bnd.N1 > Nmax, 'N1'] = Nmax
    mdf.bnd.loc[mdf.bnd.N2 > Nmax, 'N2'] = Nmax

    X1 = mdf.grd['x'][
        mdf.bnd['N1'].values.tolist(), mdf.bnd['M1'].values.tolist()]
    X2 = mdf.grd['x'][
        mdf.bnd['N2'].values.tolist(), mdf.bnd['M2'].values.tolist()]
    Y1 = mdf.grd['y'][
        mdf.bnd['N1'].values.tolist(), mdf.bnd['M1'].values.tolist()]
    Y2 = mdf.grd['y'][
        mdf.bnd['N2'].values.tolist(), mdf.bnd['M2'].values.tolist()]

    nInd = np.where((mdf.bnd['N1'] == mdf.bnd['N2']) & (X1.mask), X1, -1) + 1
    mInd = np.where((mdf.bnd['M1'] == mdf.bnd['M2']) & (X1.mask), X1, -1) + 1

    bndu = mdf.bnd.loc[:, ['M1', 'N1', 'M2', 'N2']]
    bndu.loc[:, 'N2'] -= nInd
    bndu.loc[:, 'N1'] -= nInd
    bndu.loc[:, 'M1'] -= mInd
    bndu.loc[:, 'M2'] -= mInd

    mdf.bnd['X1'] = mdf.grd['x'][
        bndu['N1'].astype(int).values.tolist(),
        bndu['M1'].astype(int).values.tolist()
    ]
    mdf.bnd['X2'] = mdf.grd['x'][
        bndu['N2'].astype(int).values.tolist(),
        bndu['M2'].astype(int).values.tolist()
    ]
    mdf.bnd['Y1'] = mdf.grd['y'][
        bndu['N1'].astype(int).values.tolist(),
        bndu['M1'].astype(int).values.tolist()
    ]
    mdf.bnd['Y2'] = mdf.grd['y'][
        bndu['N2'].astype(int).values.tolist(),
        bndu['M2'].astype(int).values.tolist()
    ]

    time = (ds.time.values - np.datetime64(mdf.Itdate)) / 60000000000.
    Nrec = len(time)
    time = time.astype(float)
    RefTime = mdf.Itdate.replace('-', '')[:8]

    for index, val in mdf.bnd.iterrows():

        BndType = val['Type']
        X1, X2, Y1, Y2 = val[['X1', 'X2', 'Y1', 'Y2']]

        D1 = np.sqrt((ds['Latitude'] - Y1)**2 + (ds['Longitude'] - X1)**2)
        D2 = np.sqrt((ds['Latitude'] - Y2)**2 + (ds['Longitude'] - X2)**2)
        ind1 = np.unravel_index(np.argmin(D1, axis=None), D1.shape)
        ind2 = np.unravel_index(np.argmin(D2, axis=None), D2.shape)

        if BndType == 'Z':

            dataA = ds['S1'][:,
                             ind1[0] - 1:ind1[0] + 2,
                             ind1[1] - 1:ind1[1] + 2]
            dataB = ds['S1'][:,
                             ind2[0] - 1:ind2[0] + 2,
                             ind2[1] - 1:ind2[1] + 2]
            prop = 'water'
        if BndType == 'C':
            dataAU = ds['u'][:,
                              ind1[0] - 1:ind1[0] + 2,
                              ind1[1] - 1:ind1[1] + 2]
            dataBU = ds['u'][:,
                              ind2[0] - 1:ind2[0] + 2,
                              ind2[1] - 1:ind2[1] + 2]
            dataAV = ds['v'][:,
                              ind1[0] - 1:ind1[0] + 2,
                              ind1[1] - 1:ind1[1] + 2]
            dataBV = ds['v'][:,
                              ind2[0] - 1:ind2[0] + 2,
                              ind2[1] - 1:ind2[1] + 2]
            prop = 'current'

        if BndType == 'R':
            pass

        if BndType != 'N':
            maskA = np.ma.masked_equal(dataA.mean('time'), 0)
            maskB = np.ma.masked_equal(dataB.mean('time'), 0)
            if maskB.ndim == 2:
                maskB = np.repeat(maskB[np.newaxis, :, :], Nrec, axis=0)
                maskA = np.repeat(maskA[np.newaxis, :, :], Nrec, axis=0)
            else:
                maskB = np.repeat(maskB[np.newaxis, :, :, :], Nrec, axis=0)
                maskA = np.repeat(maskA[np.newaxis, :, :, :], Nrec, axis=0)

            dataA = np.mean(
                np.ma.masked_array(dataA, maskA.mask), axis=(-1, -2)
            )
            dataB = np.mean(
                np.ma.masked_array(dataB, maskB.mask), axis=(-1, -2)
            )

            mdf.bct[(index, prop)]['dataA'] = dataA
            mdf.bct[(index, prop)]['dataB'] = dataB
            mdf.bct[(index, prop)]['reference-time'] = RefTime
            mdf.bct[(index, prop)]['records-in-table'] = Nrec
            mdf.bct[(index, prop)]['time'] = time
        else:
            prop = 'neumann'
            mdf.bct[(index, prop)]['dataA'] = np.array([0, 0])
            mdf.bct[(index, prop)]['dataB'] = np.array([0, 0])
            mdf.bct[(index, prop)]['reference-time'] = RefTime
            mdf.bct[(index, prop)]['records-in-table'] = 2
            mdf.bct[(index, prop)]['time'] = time[[0, -1]]

    WriteBctBcc(
        mdf.bct, os.path.join(mdf.mdfPath, mdf.Filbct.replace('.bct', '.bct'))
    )


def NestingTrim(mdf, TrimName=None, BndCoordConvert=None, NestBoundaries=None):

    if isinstance(TrimName, str):
        ds = xr.open_dataset(TrimName)
    elif isinstance(TrimName, xr.core.dataset.Dataset):
        ds = TrimName
    else:
        raise ValueError('TrimName must be a trimfilename')

    if NestBoundaries:
        if isinstance(NestBoundaries, (list, str)):
            mdf.bnd = mdf.bnd.loc[NestBoundaries, :]

    mdf.bnd.loc[:, ['M1', 'N1', 'M2', 'N2']] -= 1

    if isinstance(BndCoordConvert, (list, np.ndarray)):
        ini, end = BndCoordConvert
        mdf.grd['x'], mdf.grd['y'] = CoonvertCoordinates(
            mdf.grd['x'],
            mdf.grd['y'],
            ini,
            end
        )

    Nmax, Mmax = mdf.grd['x'].shape
    Mmax -= 1
    Nmax -= 1

    mdf.bnd.loc[mdf.bnd.M1 > Mmax, 'M1'] = Mmax
    mdf.bnd.loc[mdf.bnd.M2 > Mmax, 'M2'] = Mmax
    mdf.bnd.loc[mdf.bnd.N1 > Nmax, 'N1'] = Nmax
    mdf.bnd.loc[mdf.bnd.N2 > Nmax, 'N2'] = Nmax

    X1 = mdf.grd['x'][
        mdf.bnd['N1'].values.tolist(), mdf.bnd['M1'].values.tolist()]
    X2 = mdf.grd['x'][
        mdf.bnd['N2'].values.tolist(), mdf.bnd['M2'].values.tolist()]
    Y1 = mdf.grd['y'][
        mdf.bnd['N1'].values.tolist(), mdf.bnd['M1'].values.tolist()]
    Y2 = mdf.grd['y'][
        mdf.bnd['N2'].values.tolist(), mdf.bnd['M2'].values.tolist()]

    nInd = np.where((mdf.bnd['N1'] == mdf.bnd['N2']) & (X1.mask), X1, -1) + 1
    mInd = np.where((mdf.bnd['M1'] == mdf.bnd['M2']) & (X1.mask), X1, -1) + 1

    bndu = mdf.bnd.loc[:, ['M1', 'N1', 'M2', 'N2']]
    bndu.loc[:, 'N2'] -= nInd
    bndu.loc[:, 'N1'] -= nInd
    bndu.loc[:, 'M1'] -= mInd
    bndu.loc[:, 'M2'] -= mInd

    mdf.bnd['X1'] = mdf.grd['x'][
        bndu['N1'].astype(int).values.tolist(),
        bndu['M1'].astype(int).values.tolist()
    ]
    mdf.bnd['X2'] = mdf.grd['x'][
        bndu['N2'].astype(int).values.tolist(),
        bndu['M2'].astype(int).values.tolist()
    ]
    mdf.bnd['Y1'] = mdf.grd['y'][
        bndu['N1'].astype(int).values.tolist(),
        bndu['M1'].astype(int).values.tolist()
    ]
    mdf.bnd['Y2'] = mdf.grd['y'][
        bndu['N2'].astype(int).values.tolist(),
        bndu['M2'].astype(int).values.tolist()
    ]

    time = (ds.time.values - np.datetime64(mdf.Itdate)) / 60000000000.
    Nrec = len(time)
    time = time.astype(float)
    RefTime = mdf.Itdate.replace('-', '')[:8]

    for index, val in mdf.bnd.iterrows():

        BndType = val['Type']
        X1, X2, Y1, Y2 = val[['X1', 'X2', 'Y1', 'Y2']]

        D1 = np.sqrt((ds['YCOR'] - Y1)**2 + (ds['XCOR'] - X1)**2)
        D2 = np.sqrt((ds['YCOR'] - Y2)**2 + (ds['XCOR'] - X2)**2)
        ind1 = np.unravel_index(np.argmin(D1, axis=None), D1.shape)
        ind2 = np.unravel_index(np.argmin(D2, axis=None), D2.shape)

        if BndType == 'Z':

            dataA = ds['S1'][:,
                             ind1[0] - 1:ind1[0] + 2,
                             ind1[1] - 1:ind1[1] + 2]
            dataB = ds['S1'][:,
                             ind2[0] - 1:ind2[0] + 2,
                             ind2[1] - 1:ind2[1] + 2]
            prop = 'water'
        if BndType == 'C':
            dataAU = ds['U1'][:,
                              ind1[0] - 1:ind1[0] + 2,
                              ind1[1] - 1:ind1[1] + 2]
            dataBU = ds['U1'][:,
                              ind2[0] - 1:ind2[0] + 2,
                              ind2[1] - 1:ind2[1] + 2]
            dataAV = ds['V1'][:,
                              ind1[0] - 1:ind1[0] + 2,
                              ind1[1] - 1:ind1[1] + 2]
            dataBV = ds['V1'][:,
                              ind2[0] - 1:ind2[0] + 2,
                              ind2[1] - 1:ind2[1] + 2]
            prop = 'current'
        if BndType == 'R':
            pass

        if BndType != 'N':
            maskA = np.ma.masked_equal(dataA.mean('time'), 0)
            maskB = np.ma.masked_equal(dataB.mean('time'), 0)
            if maskB.ndim == 2:
                maskB = np.repeat(maskB[np.newaxis, :, :], Nrec, axis=0)
                maskA = np.repeat(maskA[np.newaxis, :, :], Nrec, axis=0)
            else:
                maskB = np.repeat(maskB[np.newaxis, :, :, :], Nrec, axis=0)
                maskA = np.repeat(maskA[np.newaxis, :, :, :], Nrec, axis=0)

            dataA = np.mean(
                np.ma.masked_array(dataA, maskA.mask), axis=(-1, -2)
            )
            dataB = np.mean(
                np.ma.masked_array(dataB, maskB.mask), axis=(-1, -2)
            )

            mdf.bct[(index, prop)]['dataA'] = dataA
            mdf.bct[(index, prop)]['dataB'] = dataB
            mdf.bct[(index, prop)]['reference-time'] = RefTime
            mdf.bct[(index, prop)]['records-in-table'] = Nrec
            mdf.bct[(index, prop)]['time'] = time
        else:
            prop = 'neumann'
            mdf.bct[(index, prop)]['dataA'] = np.array([0, 0])
            mdf.bct[(index, prop)]['dataB'] = np.array([0, 0])
            mdf.bct[(index, prop)]['reference-time'] = RefTime
            mdf.bct[(index, prop)]['records-in-table'] = 2
            mdf.bct[(index, prop)]['time'] = time[[0, -1]]

    WriteBctBcc(
        mdf.bct, os.path.join(mdf.mdfPath, mdf.Filbct.replace('.bct', '.bct'))
    )

def ReadBca(fname):

    bca = dict()

    f = open(fname, 'r')

    lines = f.readlines()
    #removind '\n' from lines
    lines = [line.replace('\n','') for line in lines]

    for i, line in enumerate(lines):

        if i ==0:
            p_borda = line.split()[0]
            #nome da fronteira
            bca[p_borda] = p_borda

            #dataframe das constantes
            df = pd.DataFrame(lines[i+1:i+15])
            df.columns = ['1,2,3']
            df = df['1,2,3'].str.split(expand=True)
            df.iloc[0,2] = df.iloc[0,1]

            df.index = df.iloc[:,0]
            df.drop(columns=0,inplace=True)
            df.columns = ['Amplitude','Fase']
            df=df.astype(float)

            bca[p_borda] = df

            ind = i+15

        if i == ind:
            p_borda = line.split()[0]
            #nome da fronteira
            bca[p_borda] = p_borda

            #dataframe das constantes
            df = pd.DataFrame(lines[i+1:i+15])
            df.columns = ['1,2,3']
            df = df['1,2,3'].str.split(expand=True)
            df.iloc[0,2] = df.iloc[0,1]

            df.index = df.iloc[:,0]
            df.drop(columns=0,inplace=True)
            df.columns = ['Amplitude','Fase']
            df=df.astype(float)

            bca[p_borda] = df

            ind = i+15

        else:
            pass
    return bca


def ReadBctBcc(fname):

    # gets Name
    f = open(fname, 'r')
    lines = f.readlines()
    nloc = len([line for line in lines if 'location' in line])
    data = list()
    bct = pd.DataFrame(
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
    for j, line in enumerate(lines):

        col = line.split()[0]


        if col in bct.columns:
            if "'" in line:
                bct.loc[i, col] = "".join(
                    line.split("'")[1].split('(')[0].split(':')[-1].split()[0]
                )
            else:
                bct.loc[i, col] = line.split()[-1]

        if 'records-in-table' in line:
            data.append(
                np.genfromtxt(
                    fname,
                    skip_header=j + 1,
                    max_rows=int(bct.loc[i, col])
                )
            )
            i += 1

    bct.set_index(['location', 'parameter'], drop=True, inplace=True)
    bct = bct.T.to_dict()
    for k in bct.keys():
        i = int(bct[k]['table-name'].split(':')[-1])
        bct[k]['time'] = data[i - 1][:, 0]
        n = (data[i - 1].shape[1] - 1)
        n = int(n / 2) + 1
        bct[k]['dataA'] = data[i - 1][:, 1:n]
        bct[k]['dataB'] = data[i - 1][:, n:]
        reftime = datetime.strptime(str(bct[k]['reference-time']), '%Y%m%d')
        bct[k]['datetime'] = pd.DatetimeIndex(
            data[i - 1][:, 0].astype('timedelta64[m]') + np.datetime64(reftime)
        )
    return bct


def WriteBctBcc(mdict, fname):

    f = open(fname, 'w+')
    f.close()
    for k in mdict.keys():
        f = open(fname, 'a')
        location, param = k
        f.write(
            "table-name\t'Boundary Section: {}'\n".format(
                mdict[k]['table-name']
            )
        )
        f.write("contents\t'{}'\n".format(mdict[k]['contents']))
        f.write("location\t'{}'\n".format(location))
        f.write("time-function\t'{}'\n".format(mdict[k]['time-function']))
        f.write("reference-time\t {}\n".format(mdict[k]['reference-time']))
        f.write("time-unit\t'{}'\n".format(mdict[k]['time-unit']))
        f.write("interpolation\t'{}'\n".format(mdict[k]['interpolation']))
        f.write("parameter\t'time'\tunit '[min]'\n")

        if param == 'water':
            units = '[m]'
            param = param + ' elevation (z) '
        if param == 'current':
            units = '[m/s]'
            param = param + '         (c) '
        if param == 'riemann':
            units = '[m/s]'
            param = param + '         (r) '
        if param == 'total':
            units = '[m3/s]'
            param = param + ' discharge (t) '
        elif param == 'Salinity':
            param = param + '             '
            units = '[ppt]'
        elif param == 'Temperature':
            param = param + '          '
            units = '[C]'
        elif param == 'conservativo':
            param = param + '          '
            units = '[kg/m3]'
        elif param == 'neumann':
            param = param + '         (n) '
            units = '[-]'

        if len(mdict[k]['dataA'].shape) < 2:
            n = 1
        else:
            n = mdict[k]['dataA'].shape[1]
        if n > 1:
            for i in range(n):
                layer = param + ' end A layer {}'.format(i + 1)
                f.write("parameter\t'{}' unit '{}'\n".format(layer, units))
        else:
            layer = param + ' end A'
            if 'elevation' not in param or not 'neumann':
                layer = layer + ' uniform '
            f.write("parameter\t'{}' unit '{}'\n".format(layer, units))

        if n > 1:
            for i in range(n):
                layer = param + ' end B layer {}'.format(i + 1)
                f.write("parameter\t'{}' unit '{}'\n".format(layer, units))
        else:
            layer = param + ' end B'
            if 'elevation' not in param or not 'neumann':
                layer = layer + ' uniform '
            f.write("parameter\t'{}' unit '{}'\n".format(layer, units))

        f.write("records-in-table\t{}\n".format(mdict[k]['records-in-table']))
        f.close()
        f = open(fname, 'ab')
        # np.savetxt(
        #     f,
        #     np.c_[mdict[k]['time'], mdict[k]['dataA'], mdict[k]['dataB']],
        #     fmt='%0.8f')
        np.savetxt(
            f,
            np.column_stack((mdict[k]['time'], mdict[k]['dataA'], mdict[k]['dataB'])),
            fmt='%0.8f')

        f.close()
