
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from TTutils.Utils import ConvertCoordinates
from scipy.interpolate import griddata
from geopandas import read_file


def ShpToLdb(ShapefileName, NewEpsg=None):

    gdf = read_file(ShapefileName)

    if NewEpsg:
        gdf.to_crs({'init': 'epsg:{}'.format(NewEpsg)}, inline=True)

    with open(ShapefileName.replace('.shp', '.ldb'), 'w+') as f:
        for k, item in gdf.iterrows():
            geo = item['geometry']
            f.write('F{:03d}\n'.format(k))
            if 'Polygon' in geo.geom_type:
                x, y = geo.boundary.xy
            else:
                x, y = geo.xy
            for xx, yy in zip(x, y):
                f.write(xx, yy)


def WriteXML(mdfname, xmlname=None):

    if not xmlname:
        xmlname=mdfname+'.xml'

    with open(xmlname, 'w+') as f:
        f.write('<?xml version="1.0" encoding="iso-8859-1"?>\r\n')
        f.write('<deltaresHydro xmlns="http://schemas.deltares.nl/deltaresHydro" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://schemas.deltares.nl/deltaresHydro http://content.oss.deltares.nl/schemas/d_hydro-1.00.xsd">\r\n')
        f.write('<documentation>\r\n')
        f.write('File created by    : Deltares, create_config_xml.tcl, Version 1.01\r\n')
        f.write('File creation date : 11 January 2017, 18:11:47\r\n')
        f.write('File version       : 1.00\r\n')
        f.write('</documentation>\r\n')
        f.write('<control>\r\n')
        f.write('<sequence>\r\n')
        f.write('<start>myNameFlow</start>\r\n')
        f.write('</sequence>\r\n')
        f.write('</control>\r\n')
        f.write('<flow2D3D name="myNameFlow">\r\n')
        f.write('<library>flow2d3d</library>\r\n')
        f.write('<mdfFile>{}.mdf</mdfFile>'.format(mdfname))
        f.write('</flow2D3D>\r\n')
        f.write('<delftOnline>\r\n')
        f.write('<enabled>true</enabled>\r\n')
        f.write('<urlFile>{}.url</urlFile>'.format(mdfname))
        f.write('<waitOnStart>false</waitOnStart>\r\n')
        f.write('<clientControl>true</clientControl>\r\n')
        f.write('<clientWrite>false</clientWrite>\r\n')
        f.write('</delftOnline>\r\n')
        f.write('</deltaresHydro>\r\n')
