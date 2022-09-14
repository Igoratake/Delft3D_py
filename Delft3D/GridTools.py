

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import sys
from TTutils.Seawage import SIG
from TTutils.Utils import *


def TimeStepEstimation(
        grdname,
        depname,
        layers=None,
        type='sigma',
        MaxEta=1,
        CFL=10
):
    # gets data grid
    G = ReadD3DGrid(grdname)
    # gets depth data
    gHsqrt = np.sqrt(9.8 * ReadD3DDep(depname, G))
    # if spherical change to UTM
    if 'spherical' in G['CoordinateSystem'].lower():
        epsg = 32700+SIG.DefineEPSGCode(np.mean(G['x']))
        if np.mean(G['x']) > 1:
            epsg -= 100
        G['x'], G['y'] = ConvertCoordinates(
            G['x'],
            G['y'],
            4326,
            epsg
        )
    # calcualtes spatials delta
    dxx = np.diff(G['x'], n=1, axis=0)
    dxy = np.diff(G['x'], n=1, axis=1)
    dyx = np.diff(G['y'], n=1, axis=0)
    dyy = np.diff(G['y'], n=1, axis=1)
    # calcualtes spatials delta
    distx = np.sqrt(dxx**2 + dyx**2)
    disty = np.sqrt(dxy**2 + dyy**2)
    # Calulates the minimum deltas
    DistMin = np.less_equal(disty[:-1, :], distx[:, :-1]) * disty[:-1, :] + \
        np.less(distx[:, :-1], disty[:-1, :]) * distx[:, :-1]
    # calculates deltaT
    DeltaT = CFL * DistMin / gHsqrt[:-1, :-1] / 60.
    MinDeltaT = np.min(DeltaT)
    # return
    print(u'\nBest time step : {:0.3f}'.format(MinDeltaT))
    return MinDeltaT


def Enclosure(grd, grdname):

    M = list()
    N = list()
    x, y = grd['x'], grd['y']
    n, m = x.shape
    m, n = np.meshgrid(range(m), range(n))
    m += 1
    n += 1
    m = np.ma.masked_array(m, x.mask)
    n = np.ma.masked_array(n, x.mask)
    old = [0]
    for i in range(m.shape[0]):
        mm = m[i, :].compressed()
        mm = list(set(mm) - set(old))
        mm = np.sort(mm)
        if len(mm) > 1:
            Mini = [np.ma.min(mm)]
            Mend = [np.ma.max(mm) + 1]
            d = np.diff(mm)
            if np.any(d > 1):
                inis = np.take(mm, np.where(d > 1)[0] + 1)
                Mini = Mini + inis.tolist()
                ends = np.take(mm, np.where(d > 1)[0]) + 1
                Mend = Mend + ends.tolist()
            Mini.sort()
            Mend.sort()
            for mini, mend in zip(Mini, Mend):
                M.append(mini)
                M.append(mend)
                N.append(i + 1)
                N.append(i + 1)

            old = m[i, :].compressed()
    M.append(M[0])
    N.append(N[0])
    xn = len(N)
    old = [0]
    for j in range(n.shape[1]):
        nn = n[:, j].compressed()
        nn = list(set(nn) - set(old))
        nn = np.sort(nn)
        if len(nn) > 1:
            Nini = [np.ma.min(nn)]
            Nend = [np.ma.max(nn) + 1]
            d = np.diff(nn)
            if np.any(d > 1):
                inis = np.take(nn, np.where(d > 1)[0] + 1)
                Nini = Nini + inis.tolist()
                ends = np.take(nn, np.where(d > 1)[0]) + 1
                Nend = Nend + ends.tolist()
            Nini.sort()
            Nend.sort()
            for nini, nend in zip(Nini, Nend):
                N.append(nini)
                N.append(nend)
                M.append(j + 1)
                M.append(j + 1)

            old = n[:, j].compressed()
    M.append(M[xn])
    N.append(N[xn])
    np.savetxt(grdname.replace('.grd', '.enc'), np.c_[M, N], fmt='%i')


# def WriteD3DGrid(g, fname):
#
#     with open(fname,'w+') as f:
#         f.write('Coordinate System = {}\n'.format(g['CoordinateSystem']))
#         f.write('Missing Value     =    9.99999000000000000E+00\n')
#         f.write('     {}     {}\n'.format(g['nx'],g['ny']))
#         f.write('0 0 0\n')
#         for i,x in enumerate(g['x'].shape[1])
#             f.write('ETA=    {}   '.format(i))
#             f.write('{0: <5}    '.format(i))
#             f.write('{}     {}   {}   {}   {}\n'.format())
#         # get file
#     pass


def ReadD3DGrid(fname):
    # open file
    f = open(fname, 'r')
    # read all lines
    lines = f.readlines()
    # gets projections
    k = 0
    grid = dict()
    line = lines[k]
    while 'Coordinate System' not in line:
        k += 1
        line = lines[k]
    grid['CoordinateSystem'] = line.split(' = ')[-1]
    k = k+1
    line = lines[k]
    if 'miss' in line.lower():
        k = k+1
        line = lines[k]
    grid['nx'], grid['ny'] = map(int, line.split())
    k = k+1
    f.close()

    def repfunc(x):
        x = x.replace(
            '\n',
            '').replace(
                'ETA',
                '').replace(
                    '=',
                    '').split(' ')
        x = [xx for xx in x if len(xx) > 1]
        x = [float(xx) for xx in x if not xx.isdigit()]
        if len(x) > 0:
            return x
        else:
            pass
    k = k+1
    lines = list(map(repfunc, lines[k:]))
    lines = sum(lines, [])
    # reshape grid
    lines = np.reshape(
        np.asarray(lines).flatten(),
        (grid['nx'], grid['ny'], 2),
        'F'
        )
    # mask values
    grid['x'] = np.ma.masked_equal(np.squeeze(lines[:, :, 0]), 0)
    grid['y'] = np.ma.masked_equal(np.squeeze(lines[:, :, 1]), 0)
    # return grid
    return grid


def ReadD3DDep(fname, grid, nx=None, ny=None):

    if grid:
        nx, ny = grid['nx'], grid['ny']
    dep = pd.read_table(fname, header=None, sep='\s+').values.flatten()
    dep = np.ma.masked_equal(dep, -999)
    dep = dep[~np.isnan(dep)]

    try:
        dep = np.reshape(dep, ((nx + 1), (ny + 1)),'F')
        dep = (dep[1:, 1:] + dep[:-1, 1:] + dep[1:, :-1] + dep[:-1, :-1]) / 4
    except BaseException:
        dep = np.reshape(dep, (nx, ny),'F')
    return dep


def Shp2Ldb(fname, identifycation='Name', to_epsg=None):

    gdf = gpd.read_file(fname)

    if to_epsg:
        gdf.to_crs({'init': 'epsg:' + str(to_epsg)}, inplace=True)

    if identifycation:
        gdf = gdf[[identifycation, 'geometry']]
    else:
        gdf[identifycation] = ['id_%03d' % i for i in range(gdf.shape[0])]

    f = open(fname.replace('.shp', '.ldb'), '+w')
    for _, j in gdf.iterrows():
        name, coords = j
        x, y = coords.xy

        N = len(x)
        f.write(name)
        f.write('\n')
        f.write(str(N) + ' 2\n')
        for i in range(N):
            f.write('{0:.3f} {1:.3f}\n'.format(x[i], y[i]))
            print(x[i], y[i])

    f.close()


def ReadLdb(fname):

    f = open(fname, 'r')
    lines = f.readlines()
    ini = [k + 2 for k, line in enumerate(lines) if len(line.split(' ')) == 1]
    end = [k for k, line in enumerate(lines) if len(line.split(' ')) == 1][1:]
    end.append(-1)

    def repfunc(x): return x.replace('\n', '').split(' ')
    lines = list(map(repfunc, lines))
    polygons = [np.asarray(lines[i:e]).astype(float) for i, e in zip(ini, end)]

    return polygons
