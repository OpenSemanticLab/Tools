# load measurement data from file
import re
import os 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sig


#### load files
%matplotlib widget


def i_rising_falling(x):
    """split x,y into parts determined by peaks and dips of check """
    
    rising_i_list=[]
    
    falling_i_list=[]
    
   
    i_peak_list,properties = sig.find_peaks(x)
    i_dip_list,properties = sig.find_peaks(-x)
        
    if i_peak_list[0] < i_dip_list[0]:
        ### initial rise 
        rising_i_list.append(np.arange(0,i_peak_list[0]))
                             
        ### rises between min -> max
        for i in range(len(i_peak_list)-1):
            rising_i_list.append(np.arange(i_dip_list[i],i_peak_list[i+1]))
        
        ### rise between last min -> end
        if i_dip_list[-1]<(len(x)-1):
            rising_i_list.append(np.arange(i_dip_list[-1],len(x)))
            
        ### falls between max -> min
        for i in range(len(i_dip_list)-1):
            falling_i_list.append(np.arange(i_peak_list[i],i_dip_list[i]))
            
        ### fall between last max -> end
        if i_peak_list[-1]<(len(x)-1) and i_peak_list[-1]>i_dip_list[-1]:
            falling_i_list.append(np.arange(i_peak_list[-1],len(x)))
            
        return(rising_i_list,falling_i_list)
    
    else:
        ### initial fall
        falling_i_list.append(np.arange(0,i_dip_list[0]))
                             
        ### falls between min -> max
        for i in range(len(i_dip_list)-1):
            falling_i_list.append(np.arange(i_peak_list[i],i_dip_list[i+1]))
        
        ### fall between last max -> end
        if i_peak_list[-1]<(len(x)-1) and i_peak_list[-1]>i_dip_list[-1]:
            falling_i_list.append(np.arange(i_peak_list[-1],len(x)))
            
        ### rises between max -> min
        for i in range(len(i_peak_list)):
            rising_i_list.append(np.arange(i_dip_list[i],i_peak_list[i]))
            
        ### rise between last max -> end
        if i_dip_list[-1]<(len(x)-1) and i_dip_list[-1]>i_peak_list[-1]:
            rising_i_list.append(np.arange(i_dip_list[-1],len(x)))
            
        return(rising_i_list,falling_i_list)
            
                       
def get_meta_from_header(filepath,max_header_lines = 100):
    """returns a meta dict with names and values of meatadata and a meta_d"""
    
    print(filepath)
    meta_dict={}
    meta_units_dict={}
    with open(filepath,'r', encoding = "ISO-8859-1") as f:
        for i in range(max_header_lines):
            
            line = f.readline()
            result = re.search("(\D* +) *(\d*\.*\d*) (.*)",line)
            if result is not None: 
                
                try:
                    key = result.group(1).split('  ')[0]
                    meta_dict[key] = float(result.group(2))
                    meta_units_dict[key] = result.group(3)
                except Exception as e:
                    # PRINT THE ERROR MESSAGE#
                   #print(e)
                    pass
    return(meta_dict,meta_units_dict)

def infer_number_comment_lines(filepath,delimiter=';'):
    """returns the number of comment lines for which less columns are present than in the main table"""
    ## get number of columns 
    
    with open(filepath,'r', encoding = "ISO-8859-1") as f:
        line_list = f.readlines()
        n_cols_list = [len(a.split(delimiter)) for a in line_list]
        n_cols = n_cols_list[-2] ## -2 for the case that the last line is empty
        
        
        for i in range(len(n_cols_list)):
            if n_cols_list[i]==n_cols:
                return(i)
        return(None)


def get_df_meta_dicts(filepath,delimiter = ';'):
    n_comment_lines = infer_number_comment_lines(filepath,delimiter=delimiter)
    meta_dict,meta_units_dict=get_meta_from_header(filepath,max_header_lines = n_comment_lines+1)
    df = pd.read_csv(filepath,skiprows = n_comment_lines-1,header=1,skipinitialspace=True,sep = delimiter,encoding = "ISO-8859-1" )
    return(df,meta_dict,meta_units_dict)


def elastic_modulus_and_offset(strain,stress,strain_min=0.0,strain_max =None, ax = None):
   
    i_min = np.argmin(np.abs(strain-strain_min))
    if strain_max == None: 
        i_max = len(strain)-1
    else:
        i_max = np.argmin(np.abs(strain-strain_max))
    popt = np.polyfit(strain[i_min:i_max],stress[i_min:i_max],1)
    E = popt[0]
    off = popt[1]
    if ax is not None: 
        x=strain[i_min:i_max]
        ax.plot(x*100,E*x+off,label = str(E/1000)+' [kPa]',c = 'black',linewidth=2,linestyle='--')
        ax.plot(strain*100,stress)
    return(E,off)


def plot_force_length_diagram(filepath,ax=None,show=True):
    
    """loads data from a file and plots a diagram with force vs length of all data
    if given into ax."""
    
    if ax is None:
        fig, ax = plt.subplots()
    df,meta_dict,meta_dict_units = get_df_meta_dicts(filepath)
    
    ax.plot(df['Einspannlänge'],df['Standardkraft'],label = r'Standardkraft')
    ax.set_xlabel('Einspannlänge')
    ax.set_ylabel('Standardkraft')
    ax.grid(True)
    
    
    if show and ax is None:
        fig.show()
        

    

def plot_time_diagram(filepath,show=True,mark_turning_points=False):
    """loads data from a file and plots diagrams of force and length vs time"""
    fig,(ax_force,ax_length) = plt.subplots(nrows = 2, sharex=True)
    
    df,meta_dict,meta_dict_units = get_df_meta_dicts(filepath)
    
    ax_length.plot(df['Prüfzeit'],df['Einspannlänge'])
    ax_length.set_ylabel('Einspannlänge')
    ax_length.set_xlabel('Zeit')
    ax_length.grid(True)
    
    ax_force.plot(df['Prüfzeit'],df['Standardkraft'])
    ax_force.set_ylabel('Standardkraft')
    ax_force.set_xlabel('Zeit')
    ax_force.grid(True)
    
    
    
    
    if mark_turning_points:
        i_peak_list,properties = sig.find_peaks(df['Einspannlänge'])
        i_dip_list,properties = sig.find_peaks(-df['Einspannlänge'])
       
        ax_length.scatter(df['Prüfzeit'][i_peak_list],df['Einspannlänge'][i_peak_list],color = 'red')
        ax_length.scatter(df['Prüfzeit'][i_dip_list],df['Einspannlänge'][i_dip_list],color = 'blue')
        
        i_rising_list,i_falling_list = i_rising_falling(df['Einspannlänge'])
        
        for i_rising_sublist in i_rising_list:
            ax_length.plot(df['Prüfzeit'][i_rising_sublist],df['Einspannlänge'][i_rising_sublist],
                          color = 'red',linestyle = '--')
        for i_falling_sublist in i_falling_list:
            ax_length.plot(df['Prüfzeit'][i_falling_sublist],df['Einspannlänge'][i_falling_sublist],
                          color = 'blue',linestyle = '--')
            
    
    if show:
        fig.show()

def plot_falling_and_rising(filepath):
    """loads data from a file and plots stress vs strain for all rising and fallin curves"""
    fig,(rising_ax,falling_ax) = plt.subplots(nrows=2,figsize=(6,9))
    
    rising_ax.grid(True)
    falling_ax.grid(True)
    
    df,meta_dict,meta_dict_units = get_df_meta_dicts(filepath)
    
    A = meta_dict['Probendicke']*meta_dict['Probenbreite']*1e-6
    df['lamb'] = df['Einspannlänge']/df['Einspannlänge'][0]
    df['stress'] = df['Standardkraft']/A
    
    i_rising_list,i_falling_list = i_rising_falling(df['Einspannlänge'])
    for i_rising_sublist in i_rising_list:
        rising_ax.plot(df['lamb'][i_rising_sublist]-df['lamb'][i_rising_sublist[0]],df['stress'][i_rising_sublist]*1e-6,
                        color = 'red')
    for i_falling_sublist in i_falling_list:
        falling_ax.plot(df['lamb'][i_falling_sublist]-df['lamb'][i_falling_sublist[0]],df['stress'][i_falling_sublist]*1e-6,
                        color = 'blue')
    
    
    rising_ax.set_title('rising curves')
    rising_ax.set_xlabel('relative extension')
    rising_ax.set_ylabel('engineering stress (MPa)')
    falling_ax.set_title('falling curves')
    falling_ax.set_xlabel('relative extension')
    falling_ax.set_ylabel('engineering stress (MPa)')
    
    
   # fig,(F_max_ax,min_max_ax,Young_ax) = plt.subplots(nrows=3,figsize=(6,9))
    
    
def minima_change(filepath):
    """loads data from a file and analyzes change of minia """
    df,meta_dict,meta_dict_units = get_df_meta_dicts(filepath)
    
    i_dip_list,properties = sig.find_peaks(-df['Einspannlänge'])
    
    length_change_per_cycle =[df['Einspannlänge'][i_dip_list[i+1]]-df['Einspannlänge'][i_dip_list[i-1]] for i in np.arange(1,len(i_dip_list)-1)]
    print(length_change_per_cycle)
    fig,ax = plt.subplots()
    ax.plot(length_change_per_cycle)
    ax.grid(True)
    ax.set_ylabel('length change (mm)')
    ax.set_xlabel('cycle')
    fig.show()

def main():
    folderpath = "test_data"
    print(os.listdir(folderpath))
    filenames = []
    for file in os.listdir(folderpath):
        if file.endswith("TRA"):
            filenames.append(file)
    
    filename = filenames[0]  ## files generated by Zwick export from Tetiana
    filepath = folderpath+'/'+filename
    
    df,meta_dict,meta_dict_units = get_df_meta_dicts(filepath,delimiter = ';')
    plot_force_length_diagram(filepath)
    plot_time_diagram(filepath,mark_turning_points=True)
    plot_falling_and_rising(filepath)
    minima_change(filepath)
                                  
    

if __name__ == '__main__':
    main()