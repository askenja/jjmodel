
import os
import shutil as sh
import numpy as np



def get_iso_grid(mode):
    
    if mode=='Padova' or mode=='MIST':
        folder_name = 'multiband'
    if mode=='BaSTI':
        folder_name='gaiaedr3'
        
    path = os.path.join('input','isochrones',mode,folder_name)
    
    met_grid = np.loadtxt(os.path.join('input','isochrones','Metallicity_grid.txt'))
    z_array, feh_array = met_grid.T[0], met_grid.T[1]
    
    dage = 0.05       # Gyr
    agemin = 0.05     # Gyr
    agemax = 13       # Gyr
    age_array = np.arange(agemin,agemax+dage,dage)
    
    Nfe, Nage = len(feh_array), len(age_array)
    grid_mask = np.zeros((Nfe,Nage))
    grid_mask.fill(True)
    
    folder_list = np.array(os.listdir(path))
    fe_names = [float(''.join((list(i)[6:]))) for i in folder_list]

    for i in range(Nfe):
        if np.round(feh_array[i],2) not in fe_names:
            grid_mask[i].fill(False)
        else:
            fehfolder = 'iso_fe' + str(np.round(feh_array[i],2))
            indfe = np.where(folder_list==fehfolder)[0][0]
            iso_list = os.listdir(os.path.join(path,folder_list[indfe]))
            iso_list_name = [float(''.join((list(k.split('.txt')[0])[7:]))) for k in iso_list]
            for k in range(Nage):
                if np.round(age_array[k],2) not in iso_list_name:
                    grid_mask[i,k] = False
    
    np.savetxt(os.path.join('input','isochrones',''.join(('grid_mask_',mode,'.txt'))),grid_mask)

                      

get_iso_grid('Padova')

            
    
        
    
        