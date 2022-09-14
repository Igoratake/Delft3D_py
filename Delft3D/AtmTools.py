import numpy as np
import pandas as pd
import xarray as xr

from TTutils.Utils import * 


def escreve_vento(ds,var,delta_hours,lon,lat,dlon,dlat,utm,tref,tref_file):
    
    if var == 'u':
        f = open('wind.amu','w')
        type = 'quantity1        = x_wind\n'
        vel = 'U_GRD_L103'
        
    else:
        f = open('wind.amv','w')
        type = 'quantity1        = y_wind\n'
        vel = 'V_GRD_L103'
        
    f.write('FileVersion      = 1.03\n')
    f.write('Filetype         = meteo_on_equidistant_grid\n')
    f.write('n_cols           = {}\n'.format(lat.shape[1]))
    f.write('n_rows           = {}\n'.format(lon.shape[0]))
    
    if utm == 0:
        f.write('grid_unit        = degree\n')
        x_ref = 'x_llcenter       = {}\n'.format(np.round(lon[-1,0]),2)
        y_ref = 'y_llcenter       = {}\n'.format(np.round(lat[-1,-1]),2)
    
    elif utm == 1:
        f.write('grid_unit        = m\n')
        x_ref = 'x_llcenter       = {}\n'.format(np.round(lon[0,0]),2)
        y_ref = 'y_llcenter       = {}\n'.format(np.round(lat[-1,0]),2)
        
    f.write(x_ref)
    f.write('dx               = {}\n'.format(np.round(dlon,5)))
    f.write(y_ref)
    f.write('dy               = {}\n'.format(np.round(dlat,5)))
    f.write('NODATA_value     = 999.999\n')
    f.write('n_quantity       = 1\n') 
    f.write(type)    
    f.write('unit1            = m s-1\n')
    
    for i,t in enumerate(delta_hours):
        
        print('TIME =          {} hours since {} 00:00:00 +00:00'.format(int(t),tref_file))
        f.write('TIME =          {} hours since {} 00:00:00 +00:00\n'.format(int(t),tref_file))
        
        sel = ds.isel(time=i)
        
        for y in sel.lat.values:
            sel_y = sel.sel(lat=y)
            for x in sel.lon.values:
                sel_yx = sel_y.sel(lon=x)
                
                value = np.round(sel_yx[vel].values,3)
                
                if x != sel.lon.values[-1]:
                   f.write(' {v:=1.3f} '.format(v=value))
                else:
                   f.write(' {v:=1.3f} \n'.format(v=value))
    
    f.close()    
    
    return None

def escreve_pres(ds,delta_hours,lon,lat,dlon,dlat,utm,tref,tref_file):
    
    #var = 'PRES_L1'
    var = 'PRMSL_L101'


    f = open('wind.amp','w')  
        
    f.write('FileVersion      = 1.03\n')
    f.write('Filetype         = meteo_on_equidistant_grid\n')
    f.write('n_cols           = {}\n'.format(lat.shape[1]))
    f.write('n_rows           = {}\n'.format(lon.shape[0]))
    
    if utm == 0:
        f.write('grid_unit        = degree\n')
        x_ref = 'x_llcenter       = {}\n'.format(np.round(lon[-1,0]),2)
        y_ref = 'y_llcenter       = {}\n'.format(np.round(lat[-1,-1]),2)
    
    elif utm == 1:
        f.write('grid_unit        = m\n')
        x_ref = 'x_llcenter       = {}\n'.format(np.round(lon[0,0]),2)
        y_ref = 'y_llcenter       = {}\n'.format(np.round(lat[-1,0]),2)
        
    f.write(x_ref)
    f.write('dx               = {}\n'.format(np.round(dlon,5)))
    f.write(y_ref)
    f.write('dy               = {}\n'.format(np.round(dlat,5)))
    f.write('NODATA_value     = 999.999\n')
    f.write('n_quantity       = 1\n') 
    f.write('quantity1        = air_pressure\n')    
    f.write('unit1            = Pa\n')
    
    for i,t in enumerate(delta_hours):
        
        print('TIME =          {} hours since {} 00:00:00 +00:00'.format(int(t),tref_file))
        f.write('TIME =          {} hours since {} 00:00:00 +00:00\n'.format(int(t),tref_file))
        
        sel = ds.isel(time=i)
        
        for y in sel.lat.values:
            sel_y = sel.sel(lat=y)
            for x in sel.lon.values:
                sel_yx = sel_y.sel(lon=x)
                
                value = np.round(sel_yx[var].values,3)
                
                if x != sel.lon.values[-1]:
                   f.write(' {v:=1.3f} '.format(v=value))
                else:
                   f.write(' {v:=1.3f} \n'.format(v=value))
    
    f.close()    
    
    return None

def escreve_amr(ds,delta_hours,lon,lat,dlon,dlat,utm,tref,tref_file):
    
    #var = 'PRES_L1'
    var = 'R_H_L103'


    f = open('wind.amr','w')  
        
    f.write('FileVersion      = 1.03\n')
    f.write('Filetype         = meteo_on_equidistant_grid\n')
    f.write('n_cols           = {}\n'.format(lat.shape[1]))
    f.write('n_rows           = {}\n'.format(lon.shape[0]))
    
    if utm == 0:
        f.write('grid_unit        = degree\n')
        x_ref = 'x_llcenter       = {}\n'.format(np.round(lon[-1,0]),2)
        y_ref = 'y_llcenter       = {}\n'.format(np.round(lat[-1,-1]),2)
    
    elif utm == 1:
        f.write('grid_unit        = m\n')
        x_ref = 'x_llcenter       = {}\n'.format(np.round(lon[0,0]),2)
        y_ref = 'y_llcenter       = {}\n'.format(np.round(lat[-1,0]),2)
        
    f.write(x_ref)
    f.write('dx               = {}\n'.format(np.round(dlon,5)))
    f.write(y_ref)
    f.write('dy               = {}\n'.format(np.round(dlat,5)))
    f.write('NODATA_value     = 999.999\n')
    f.write('n_quantity       = 1\n') 
    f.write('quantity1        = relative_humidity\n')    
    f.write('unit1            = %\n')
    
    for i,t in enumerate(delta_hours):
        
        print('TIME =          {} hours since {} 00:00:00 +00:00'.format(int(t),tref_file))
        f.write('TIME =          {} hours since {} 00:00:00 +00:00\n'.format(int(t),tref_file))
        
        sel = ds.isel(time=i)
        
        for y in sel.lat.values:
            sel_y = sel.sel(lat=y)
            for x in sel.lon.values:
                sel_yx = sel_y.sel(lon=x)
                
                value = np.round(sel_yx[var].values,3)
                
                if x != sel.lon.values[-1]:
                   f.write(' {v:=1.3f} '.format(v=value))
                else:
                   f.write(' {v:=1.3f} \n'.format(v=value))
    
    f.close()    
    
    return None

def escreve_amc(ds,delta_hours,lon,lat,dlon,dlat,utm,tref,tref_file):
    
    #var = 'PRES_L1'
    var = 'T_CDC_L200_Avg_1'


    f = open('wind.amc','w')  
        
    f.write('FileVersion      = 1.03\n')
    f.write('Filetype         = meteo_on_equidistant_grid\n')
    f.write('n_cols           = {}\n'.format(lat.shape[1]))
    f.write('n_rows           = {}\n'.format(lon.shape[0]))
    
    if utm == 0:
        f.write('grid_unit        = degree\n')
        x_ref = 'x_llcenter       = {}\n'.format(np.round(lon[-1,0]),2)
        y_ref = 'y_llcenter       = {}\n'.format(np.round(lat[-1,-1]),2)
    
    elif utm == 1:
        f.write('grid_unit        = m\n')
        x_ref = 'x_llcenter       = {}\n'.format(np.round(lon[0,0]),2)
        y_ref = 'y_llcenter       = {}\n'.format(np.round(lat[-1,0]),2)
        
    f.write(x_ref)
    f.write('dx               = {}\n'.format(np.round(dlon,5)))
    f.write(y_ref)
    f.write('dy               = {}\n'.format(np.round(dlat,5)))
    f.write('NODATA_value     = 999.999\n')
    f.write('n_quantity       = 1\n') 
    f.write('quantity1        = cloudiness\n')    
    f.write('unit1            = %\n')
    
    for i,t in enumerate(delta_hours):
        
        print('TIME =          {} hours since {} 00:00:00 +00:00'.format(int(t),tref_file))
        f.write('TIME =          {} hours since {} 00:00:00 +00:00\n'.format(int(t),tref_file))
        
        sel = ds.isel(time=i)
        
        for y in sel.lat.values:
            sel_y = sel.sel(lat=y)
            for x in sel.lon.values:
                sel_yx = sel_y.sel(lon=x)
                
                value = np.round(sel_yx[var].values,3)
                
                if x != sel.lon.values[-1]:
                   f.write(' {v:=1.3f} '.format(v=value))
                else:
                   f.write(' {v:=1.3f} \n'.format(v=value))
    
    f.close()    
    
    return None

def escreve_amt(ds,delta_hours,lon,lat,dlon,dlat,utm,tref,tref_file):
    
    #var = 'PRES_L1'
    var = 'Celsius'


    f = open('wind.amt','w')  
        
    f.write('FileVersion      = 1.03\n')
    f.write('Filetype         = meteo_on_equidistant_grid\n')
    f.write('n_cols           = {}\n'.format(lat.shape[1]))
    f.write('n_rows           = {}\n'.format(lon.shape[0]))
    
    if utm == 0:
        f.write('grid_unit        = degree\n')
        x_ref = 'x_llcenter       = {}\n'.format(np.round(lon[-1,0]),2)
        y_ref = 'y_llcenter       = {}\n'.format(np.round(lat[-1,-1]),2)
    
    elif utm == 1:
        f.write('grid_unit        = m\n')
        x_ref = 'x_llcenter       = {}\n'.format(np.round(lon[0,0]),2)
        y_ref = 'y_llcenter       = {}\n'.format(np.round(lat[-1,0]),2)
        
    f.write(x_ref)
    f.write('dx               = {}\n'.format(np.round(dlon,5)))
    f.write(y_ref)
    f.write('dy               = {}\n'.format(np.round(dlat,5)))
    f.write('NODATA_value     = 999.999\n')
    f.write('n_quantity       = 1\n') 
    f.write('quantity1        = air_temperature\n')    
    f.write('unit1            = Celsius\n')
    
    for i,t in enumerate(delta_hours):
        
        print('TIME =          {} hours since {} 00:00:00 +00:00'.format(int(t),tref_file))
        f.write('TIME =          {} hours since {} 00:00:00 +00:00\n'.format(int(t),tref_file))
        
        sel = ds.isel(time=i)
        
        for y in sel.lat.values:
            sel_y = sel.sel(lat=y)
            for x in sel.lon.values:
                sel_yx = sel_y.sel(lon=x)
                
                value = np.round(sel_yx[var].values,3)
                
                if x != sel.lon.values[-1]:
                   f.write(' {v:=1.3f} '.format(v=value))
                else:
                   f.write(' {v:=1.3f} \n'.format(v=value))
    
    f.close()    

def escreve_ams(ds,delta_hours,lon,lat,dlon,dlat,utm,tref,tref_file):
    
    #var = 'PRES_L1'
    var = 'DSWRF_L1_Avg_1'


    f = open('wind.ams','w')  
        
    f.write('FileVersion      = 1.03\n')
    f.write('Filetype         = meteo_on_equidistant_grid\n')
    f.write('n_cols           = {}\n'.format(lat.shape[1]))
    f.write('n_rows           = {}\n'.format(lon.shape[0]))
    
    if utm == 0:
        f.write('grid_unit        = degree\n')
        x_ref = 'x_llcenter       = {}\n'.format(np.round(lon[-1,0]),2)
        y_ref = 'y_llcenter       = {}\n'.format(np.round(lat[-1,-1]),2)
    
    elif utm == 1:
        f.write('grid_unit        = m\n')
        x_ref = 'x_llcenter       = {}\n'.format(np.round(lon[0,0]),2)
        y_ref = 'y_llcenter       = {}\n'.format(np.round(lat[-1,0]),2)
        
    f.write(x_ref)
    f.write('dx               = {}\n'.format(np.round(dlon,5)))
    f.write(y_ref)
    f.write('dy               = {}\n'.format(np.round(dlat,5)))
    f.write('NODATA_value     = 999.999\n')
    f.write('n_quantity       = 1\n') 
    f.write('quantity1        = sw_radiation_flux\n')    
    f.write('unit1            = W/m2\n')
    
    for i,t in enumerate(delta_hours):
        
        print('TIME =          {} hours since {} 00:00:00 +00:00'.format(int(t),tref_file))
        f.write('TIME =          {} hours since {} 00:00:00 +00:00\n'.format(int(t),tref_file))
        
        sel = ds.isel(time=i)
        
        for y in sel.lat.values:
            sel_y = sel.sel(lat=y)
            for x in sel.lon.values:
                sel_yx = sel_y.sel(lon=x)
                
                value = np.round(sel_yx[var].values,3)
                
                if x != sel.lon.values[-1]:
                   f.write(' {v:=1.3f} '.format(v=value))
                else:
                   f.write(' {v:=1.3f} \n'.format(v=value))
    
    f.close()    

'''exemplo escreve vento

coord_dat = 4326
coord_mod = 32723 # Inserir o epsg da coordenada desejada. UTM Sul

file = 'O18041_cfsr_daymean.nc' #arquivo unico. é preciso concatenar os ncs do CFSR.
tref = '01-01-2019'
tref_file = '2019-01-01'

vel_u = 'U_GRD_L103'
vel_v = 'V_GRD_L103'

if __name__ == '__main__':

    wnd = xr.open_dataset(file)
    
    #delta de tempo
    time = pd.to_datetime(wnd.time.values)    
    tref_datetime = pd.to_datetime(tref)    
    delta_hours = ((time-tref_datetime).total_seconds())/3600
        
    #coordenadas    
    if wnd.lon.values[0] > 180:
        wnd['lon'] = wnd.lon - 360   
    
    #meshgrid distancias
    lon, lat = np.meshgrid(wnd.lon.values, wnd.lat.values)   
        
    if coord_mod != 0:
        
        lon, lat = ConvertCoordinates(lon, lat, coord_dat, coord_mod)
        utm = 1 
    
    else:
        utm = 0
    
    #delta distancia
    dlon = lon[0,1] - lon[0,0]
    dlat = lat[0,0] - lat[1,0]
        
    for var in ['u','v']:
        
        escreve_vento(var,delta_hours,lon,lat,dlon,dlat,utm)      
        
'''