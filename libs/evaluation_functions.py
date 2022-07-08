import numpy as np
import math
import scipy.constants as const
import scipy.signal as sig
from scipy.interpolate import interp1d
import lmfit
from lmfit import Parameters
import matplotlib.pyplot as plt
import pandas as pd
import copy

try: 
    from LAP_eval import filehandling as fh
except: 
    import filehandling as fh


    
def remove_cosmics(intensity,factor=1.2):
    intensity=np.array(intensity)
    intensityx = np.append(intensity,[intensity[-1],intensity[-1]])
    intensity1 = np.append(intensity[0],intensityx)
    fraction=intensity[1::]/intensity[0:-1]
    test = np.where(fraction<factor,intensity[1::],(intensity1[0:-4]+intensity1[4::])/2)
    temp=np.insert(test,0,intensity[0],axis=0)
    intensityx = np.append(intensity[0],intensity)
    intensityy = np.append(intensityx,intensity[-1])
    intensity1 = np.append(intensity[0],intensityy)
    fraction2=intensity[0:-1]/intensity[1::]
    return np.insert(np.where(fraction2<factor,temp[0:-1],(intensity1[0:-4]+intensity1[4::])/2),-1,intensity[-1],axis=0)



def plot_brute_leastsquares_results(result, best_vals=True, varlabels=None,
                       output=None, leastsq_fit_result=None):
    """Visualize the result of the brute force grid search.

    The output file will display the chi-square value per parameter and contour
    plots for all combination of two parameters.

    Inspired by the `corner` package (https://github.com/dfm/corner.py).

    Parameters
    ----------
    result : :class:`~lmfit.minimizer.MinimizerResult`
        Contains the results from the :meth:`brute` method.

    best_vals : bool, optional
        Whether to show the best values from the grid search (default is True).

    varlabels : list, optional
        If None (default), use `result.var_names` as axis labels, otherwise
        use the names specified in `varlabels`.

    output : str, optional
        Name of the output PDF file (default is 'None')
    """
    from matplotlib.colors import LogNorm
    import matplotlib.cm as cm
    
    npars = len(result.var_names)
    _fig, axes = plt.subplots(npars, npars)

    if not varlabels:
        varlabels = result.var_names
    if best_vals and isinstance(best_vals, bool):
        best_vals = result.params
    if leastsq_fit_result is not None:
        best_vals_leastsq = leastsq_fit_result.params

    for i, par1 in enumerate(result.var_names):
        for j, par2 in enumerate(result.var_names):

            # parameter vs chi2 in case of only one parameter
            if npars == 1:
                axes.plot(result.brute_grid, result.brute_Jout, 'o', ms=3)
                axes.set_ylabel(r'$\chi^{2}$')
                axes.set_xlabel(varlabels[i])
                if best_vals:
                    axes.axvline(best_vals[par1].value, ls='dashed', color='b')
                if best_vals_leastsq:
                    axes.axvline(best_vals_leastsq[par1].value, ls='dashed', color='r')

            # parameter vs chi2 profile on top
            elif i == j and j < npars-1:
                if i == 0:
                    axes[0, 0].axis('off')
                ax = axes[i, j+1]
                red_axis = tuple([a for a in range(npars) if a != i])
                ax.plot(np.unique(result.brute_grid[i]),
                        np.minimum.reduce(result.brute_Jout, axis=red_axis),
                        'o', ms=3)
                ax.set_ylabel(r'$\chi^{2}$')
                ax.yaxis.set_label_position("right")
                ax.yaxis.set_ticks_position('right')
                ax.set_xticks([])
                if best_vals:
                    ax.axvline(best_vals[par1].value, ls='dashed', color='b')
                if best_vals_leastsq:
                    ax.axvline(best_vals_leastsq[par1].value, ls='dashed', color='r')
            # parameter vs chi2 profile on the left
            elif j == 0 and i > 0:
                ax = axes[i, j]
                red_axis = tuple([a for a in range(npars) if a != i])
                ax.plot(np.minimum.reduce(result.brute_Jout, axis=red_axis),
                        np.unique(result.brute_grid[i]), 'o', ms=3)
                ax.invert_xaxis()
                ax.set_ylabel(varlabels[i])
                if i != npars-1:
                    ax.set_xticks([])
                else:
                    ax.set_xlabel(r'$\chi^{2}$')
                if best_vals:
                    ax.axhline(best_vals[par1].value, ls='dashed', color='b')
                if best_vals_leastsq:
                    ax.axhline(best_vals_leastsq[par1].value, ls='dashed', color='r')

            # contour plots for all combinations of two parameters
            elif j > i:
                ax = axes[j, i+1]
                red_axis = tuple([a for a in range(npars) if a not in (i, j)])
                X, Y = np.meshgrid(np.unique(result.brute_grid[i]),
                                   np.unique(result.brute_grid[j]))
                lvls1 = np.linspace(result.brute_Jout.min(),
                                    np.median(result.brute_Jout)/2.0, 20, dtype='int')
                lvls2 = np.linspace(np.median(result.brute_Jout)/2.0,
                                    np.median(result.brute_Jout), 20, dtype='int')
                
                lvls = np.unique(np.concatenate((lvls1, lvls2)))
                
                ax.contourf(X.T, Y.T, np.minimum.reduce(result.brute_Jout, axis=red_axis),
                            lvls, norm=LogNorm(),
                            cmap=cm.nipy_spectral)
                ax.set_yticks([])
                if best_vals:
                    ax.axvline(best_vals[par1].value, ls='dashed', color='b')
                    ax.axhline(best_vals[par2].value, ls='dashed', color='b')
                    ax.plot(best_vals[par1].value, best_vals[par2].value, 'bs', ms=3)
                if best_vals_leastsq:
                    ax.axvline(best_vals_leastsq[par1].value, ls='dashed', color='r')
                    ax.axhline(best_vals_leastsq[par2].value, ls='dashed', color='r')
                    ax.plot(best_vals_leastsq[par1].value, best_vals_leastsq[par2].value, 'rs', ms=3)
                if j != npars-1:
                    ax.set_xticks([])
                else:
                    ax.set_xlabel(varlabels[i])
                if j - i >= 2:
                    axes[i, j].axis('off')

    if output is not None:
        plt.savefig(output)
    else: 
        plt.tight_layout()
        plt.show()
   

def leastsq_fit(fun, x_data, y_data,p_names=None,p0_dict=None,weight_data=None,p_min_max_steps_dict=None,const_params=[]):
    
    ''' A wrapper around the leastsq fit from LMFIT
    fun is a function of the form fun(x_data,p0,p1,p2,...)
    '''
    
    params = Parameters() ### initialize LMfit parameters
    for p_name in p_names:
        min_val=p_min_max_steps_dict[p_name][0]
        max_val=p_min_max_steps_dict[p_name][1]
        steps=p_min_max_steps_dict[p_name][2]
        if p0_dict is not None: 
            value=p0_dict[p_name]
        params.add(p_name,value=min_val,
                   min=value,
                   max=max_val)#,
                   #brute_step=(max_val-min_val)/(steps-1))
    
    def minimize_fun(pars):
        print(pars)
        v=pars.valuesdict()
        arglist=[]
        for p_name in p_names:
            arglist.append(v[p_name])
        
        for const_param in const_params:
            arglist.append(const_param)
        
        ret=np.array((fun(x_data,*arglist)-y_data),dtype=float)
        if weight_data is not None:
            ret=ret*np.sqrt(weight_data)
        return(ret)
    conv=1E-18
    result = lmfit.minimize(minimize_fun, params=params,method='leastsq',nan_policy='omit',xtol=conv,ftol=conv,max_nfev=1000)
    print(lmfit.fit_report(result))
    arg_list=[]
    for p_name in p_names:
        arg_list.append(result.params.valuesdict()[p_name])
    
    return(arg_list)

def brute_leastsquare_fit(fun, x_data, y_data,weight_data=None,p_names=None,p_min_max_steps_dict=None,
                          const_params=[], visualize=False):
    
    """A very robust fit routine inspired from 
    https://lmfit.github.io/lmfit-py/examples/example_brute.html
    that first performs a brute force fit with subsequent least squares fit of 
    best results"""
    
    if p_names == None or p_min_max_steps_dict==None:
        raise Exception ('p_names and p_min_max_steps must be given!'+ 
                         'structure of p_min_max_steps_dict: {"pname0":[min0,max0,brute_steps0]}')
   
    params = Parameters() ### initialize LMfit parameters
    for p_name in p_names:
        min_val=p_min_max_steps_dict[p_name][0]
        max_val=p_min_max_steps_dict[p_name][1]
        steps=p_min_max_steps_dict[p_name][2]
        params.add(p_name,value=min_val,
                   min=min_val,
                   max=max_val,
                   brute_step=(max_val-min_val)/(steps-1))
        
    ### define function to be minimized for fit 
    
    def cost_function_fit(p=params):
            def minimize_fun(pars):
                
                v=pars.valuesdict()
                arglist=[]
                for p_name in p_names:
                    arglist.append(v[p_name])
                
                for const_param in const_params:
                    arglist.append(const_param)
                
                ret=np.array((fun(x_data,*arglist)-y_data),dtype=float)
                if weight_data is not None:
                    ret=ret*np.sqrt(weight_data)
                return(ret)
            brute_result=lmfit.minimize(minimize_fun,params,method='brute',nan_policy='omit')
            best_result=copy.deepcopy(brute_result)
            for candidate in brute_result.candidates[0:5]:
                trial = lmfit.minimize(minimize_fun, params=candidate.params,method='leastsq',nan_policy='omit')
                if trial.chisqr < best_result.chisqr:
                    best_result = trial
            
            return((best_result,brute_result))
            
    best_result,brute_result = cost_function_fit()
    arg_list=[]
    for p_name in p_names:
        arg_list.append(best_result.params.valuesdict()[p_name])
    for const_param in const_params:
        arg_list.append(const_param)
    
        
    if visualize == True:
        plot_brute_leastsquares_results(brute_result,leastsq_fit_result=best_result)
        plt.figure()
        plt.plot(x_data,y_data,label='data',color='blue')
        plt.plot(x_data,fun(x_data,*arg_list),label='Fit',color='red')
        plt.title(best_result.params.valuesdict())
        plt.show()
    return (arg_list[0:len(p_names)])
     
def stitchSpectra(lamb_list,count_list, method="scale", edgeremove=(0, 0), shiftToPositive=False, dlambda=None):
    """
    Stitches the raw spectra together. For this purpose, the following spectra are shifted
    to match the previous spectra in the overlapping region. Afterwards the whole data
    is interpolated on a fixed grid.

    :param str method: stitching method (possible values: scale, shift)
    :param edgeremove: ratio of omitted data at the edges
        (e.g. (0.05, 0.05) and edgetype="symmetric": first 5% and last 5% of data is omitted)
    :type edgeremove: tuple(float, float)
    :param bool shiftToPositive: if True the spectrum is shifted such that min(spectrum) >= 0
    :param float dlambda: custom wavelength steps for interpolation, None for default
    """
    rawData=np.array([np.array(lamb_list),np.array(count_list)])
    rawData=rawData.swapaxes(0,1)
    coefficients = []
    print("Removing edges for stitching:", *edgeremove)
    omitRight = rawData[0].shape[1] - math.floor(rawData[0].shape[1] * edgeremove[1])
    print("Stitching index range is ", 0, omitRight)
    processed = np.array(rawData[0][:, 0:omitRight])  
    if dlambda is None:
        dlambda = math.fabs(processed[0, 1] - processed[0, 0])  ## lambda steps of first spectrum are kept
    for i, spec in enumerate(rawData[1:]):
        omitLeft = math.floor(spec.shape[1] * edgeremove[0])
        omitRight = spec.shape[1] - math.floor(spec.shape[1] * edgeremove[1])
        print("Stitching index range is ", omitLeft, omitRight)
        if i == len(rawData)-2:
            spec = np.array(spec[:, omitLeft:])  ## do not shorten last array at end
        else:
            spec = np.array(spec[:, omitLeft:omitRight]) # shorten middle arrays at both sides
        print("Stitching spectrum in range", np.min(spec[0,]), np.max(spec[0,]))
        # calculate overlap
        overlap = (np.min(spec[0,]), np.max(processed[0,])) 
        #lambdas = np.arange(*overlap, dlambda)
        #leftfun = interp1d(processed[0,], processed[1,])
        #rightfun = interp1d(spec[0,], spec[1,])
        left = np.mean(processed[1, processed[0,] > overlap[0]]) ##mean of counts of overlap
        right = np.mean(spec[1, spec[0,] < overlap[1]])
        if method == "shift":
            # calculate offset in overlap region
            offset = left - right
            print("Stitching offset %s in overlap", offset, *overlap)
            # add shifted spectrum
            spec[1,] = spec[1,] + offset
            coefficients.append(offset)
        elif method == "scale":
            # calculate factor in overlap region
            factor = left/right
            print("Stitching factor"+str(factor)+"  in overlap ", *overlap)
            spec[1,] = spec[1,] * factor
            coefficients.append(factor)
        processed = np.concatenate([processed, spec], axis=1)
    # interpolate data on grid
    interpolated = interp1d(processed[0,], processed[1,])
    lambdas = np.arange(processed[0, 0], processed[0, -1], dlambda)
    specdata = interpolated(lambdas)
    # shift above 0
    if shiftToPositive:
        minimum = np.min(specdata)
        if minimum < 0:
            specdata += math.fabs(minimum)
    
    return (lambdas,specdata,coefficients)


def get_calibration_wl_intensity_from_pd (df,wl_string='wavelength',counts_string='counts',n_peaks_max=3,save_string=''):
    
    """calculates a calibration spectrum from multiple Fabry-Perot reflection specta"""
    
    counts_array=[]
    
    for i in df.index:
        counts_array.append(np.array(df['data'][i][counts_string]))
    counts_array=np.array(counts_array)
    
    wl_list=df['data'][0][wl_string]
    
    trans=counts_array.transpose()
    
    win=sig.hann(20)
    counts_norm=[]
    
    for i in range(len( wl_list)):
        
        filtered=sig.convolve(trans[i],win,mode='same')
        i_peak,dump = sig.find_peaks(filtered)
        counts_norm.append(np.sum(trans[i][i_peak[0]:i_peak[n_peaks_max]])/(i_peak[n_peaks_max]-i_peak[0]))
       # print(lamb_list[0][i],i_peak[1],i_peak[2])
    counts_norm=np.array(counts_norm)
    if save_string!='':
        headers=[wl_string,counts_string]
        lists=[wl_list,counts_norm]
        fh.save_headers_lists_to_csv(headers,lists,save_string,commentlines=3)
    return (wl_list,counts_norm)



def distance_from_spec(lamb,intensity,calib_spec_path,return_dict=False,plot=False,input_error=False,d_force=0,theta_max=0,d_fit_min=1e-9,d_fit_max=10e-6,refractive_index=2.6,function='none',intensity_calib_override='none',intensity_override='none'):
    
    
    """Distance_from_spec calculates a distance from a reflectance spectrum.
    Lamb is a list of wavelengths
    Intensity is a list of counts
    calib_spec_path is a path to the calibration spectrum potentially calculated by the function get_calibration_wl_intensity_from_pd
    theta_max is the maximum opening angle
    assumes symmetric cavity, refractive_index=2.6 is default for SiC
    
    Returns d if return_dict is not True
    Returns a dictionary with d, lamb_normed, func_vals.
    """

    lamb=np.array(lamb)
    intensity=np.array(intensity)
    
    try:
        calib_data=pd.read_csv(calib_spec_path,names=['wavelength','counts'],skiprows=4,delimiter='\t')
    except: 
        print('Error: Could not load datafile')
    
    lamb_calib,intensity_calib= np.array(calib_data['wavelength']),np.array(calib_data['counts'])
    
    if intensity_calib_override != 'none':
        intensity_calib=intensity_calib_override
    
    ### Find minima and maxima
    
    lamb_min=min(lamb[0],lamb_calib[0])
    lamb_max=max(lamb[-1],lamb_calib[-1])
    
    i_min_lamb=np.argmin(np.abs(lamb-lamb_min))
    i_min_lamb_calib=np.argmin(np.abs(lamb_calib-lamb_min))
    i_max_lamb=np.argmin(np.abs(lamb-lamb_max))+1
    i_max_lamb_calib=np.argmin(np.abs(lamb_calib-lamb_max))+1
    
    
    I_normed=intensity[i_min_lamb:i_max_lamb]/intensity_calib[i_min_lamb_calib:i_max_lamb_calib]
    if intensity_override!='none':
        I_normed=intensity_override
    lamb_normed=lamb[i_min_lamb:i_max_lamb]

    
    ### Possible Fit-functions      
    
    def FabryPerot(d,F,lamb):
        """The Fabry-Perot reflectance formula"""
        return(1-1/(1+F*np.sin(2*np.pi*d/lamb)**2))
    
    def FabryPerot_min_max(d,I_min,I_max,F,lamb):
        """The Fabry_perot reflectance formula mapped to I_min to I_max"""
    
        return(I_min+(I_max-I_min)/(1-1/(1+F))*(1-1/(1+F*np.sin(2*np.pi*d/lamb)**2)))
        
    
        
    def FabryPerot_min_max_theta_max(d,I_min,I_max,F,lamb,theta_max):
        """The Fabry_perot reflectance formula including finite opening angle theta_max mapped to I_min to I_max"""
        R_list=0*lamb
        theta_list=np.linspace(0,theta_max,10)
        for theta in theta_list:
            R_list=R_list+FabryPerot(d*np.cos(theta),F,lamb)*np.sin(theta)#*np.cos(theta)**2
        
        Delta=I_max-I_min
        base=np.min(R_list)
        var=R_list-base
        R_list=I_min+var*Delta/max(var)
        return(R_list)
        
    ### do Fabry Perot fit the method is analogous to the brute-least-squares fit from lap_eval.evaluation_functions
    params=Parameters()
    
    print('d_fit_min,d_fit_max',d_fit_min,d_fit_max)
    params.add('d',d_fit_min,min=d_fit_min,max=d_fit_max,brute_step=2e-9)
    print('d_start' , d_fit_min)
    I_min=np.percentile(I_normed,1) ## I_min=np.percentile(I_normed,5)
    I_max=np.percentile(I_normed,97) ## I_min=np.percentile(I_normed,95)
    
    def R (n):
        return(((1-n)/(1+n))**2)
               
    def F_symm(R_0):
        return(4*R_0/(1-R_0)**2)
        
    F=F_symm(R(refractive_index))
    

    #FabryPerot_min_max_theta_max=np.frompyfunc(FabryPerot_min_max_theta_max,6,1)    
    
        
    def cost_function_fit(lamb_list,I_list,I_min,I_max,F,params=params):
        def fun(pars):
            parvals = pars.valuesdict()
            d=parvals['d']
            
            if theta_max==0:
            
                if function=='none':
                    ret=np.array((FabryPerot_min_max(d,I_min,I_max,F,lamb_list)-I_list)**2,dtype=float)
                else:
                    ret=np.array((function(d*1e-9,lamb_list*1e-9)-I_list)**2,dtype=float)
            else:
                
                if function == 'none':
                    R_list = FabryPerot_min_max_theta_max(d,I_min,I_max,F,lamb_list,theta_max)
                    ret = np.array(R_list-I_list,dtype=float)
                else:
                    print('function not implemented with maximum angle')
                
            return(ret)
        brute_result=lmfit.minimize(fun,params,method='brute')
        best_result=copy.deepcopy(brute_result)
        for candidate in brute_result.candidates[0:5]:
            trial = lmfit.minimize(fun, params=candidate.params,method='leastsq')
            if trial.chisqr < best_result.chisqr:
                best_result = trial
        return(best_result.params.valuesdict())
    
    fit_dict=cost_function_fit(lamb_normed,I_normed*np.linspace(1,1.0,len(I_normed)),I_min,I_max,F,params)
    print(fit_dict)
    d=fit_dict['d']
    print('return_1_start_'+str(d)+'_return_1_stop')
   
    
    if(plot):
        
        fig,((ax1,ax2),(ax3,ax4))= plt.subplots(nrows=2,ncols=2)
            
        
        ax1.plot(lamb*1e9,intensity)
        ax1.set_xlabel(r'$\lambda$ in nm')
        ax1.set_ylabel('intensity in a.u.')
        ax1.set_title('raw data')
       
        ax2.plot(lamb_calib*1e9,intensity_calib)
        ax2.set_xlabel(r'$\lambda$ in nm')
        ax2.set_ylabel('intensity in a.u.')
        ax2.set_title('calibration data')
        
        ax3.plot(lamb_normed*1e9,I_normed)
        ax3.set_xlabel(r'$\lambda$ in nm')
        ax3.set_ylabel('normed intensity in a.u.')
        
        
        
        if(input_error):
            ax3.set_title('Input error: Spectrum of test data: d= '+'{d:.0f} nm'.format(d=d),color='red')
        else:
            ax3.set_title('calibrated spectrum: d= '+'{d:.0f} nm'.format(d=d*1e9))
 
        
        ax4.plot(const.c/lamb_normed/1e12,I_normed)
        ax4.set_xlabel(r'$f$ in THz')
        ax4.set_ylabel('normed intensity in a.u.')
        ax1.figure.tight_layout()
        ax1.figure.canvas.draw()
        
        if function == 'none':
            if theta_max == 0 : 
                ax3.plot(lamb_normed*1e9,FabryPerot_min_max(fit_dict['d'],I_min,I_max,F,lamb_normed))
            else: 
                ax3.plot(lamb_normed*1e9,FabryPerot_min_max_theta_max(fit_dict['d'],I_min,I_max,F,lamb_normed,theta_max))
                
        else:
            ax3.plot(lamb_normed,function(d*1e-9,lamb_normed*1e-9))
      
            
    if plot:
        plt.show()
    print('fitted distance: ',d)
    
    if(return_dict):
        func_vals=FabryPerot_min_max_theta_max(fit_dict['d'],I_min,I_max,F,lamb_normed,theta_max)
        print(func_vals)
        return({'d':d,'lamb_normed':lamb_normed,'func_vals':func_vals})
    else:            
        return(d)



if __name__=='__main__':
    ## generate some example data for fitting and testing routines 
    test_brute_fit=True
    if test_brute_fit:
        def fun(x,a,b,c):
            ret=a*np.sin(b*x)*np.exp(c*x)
            return(ret)
            
        df=pd.DataFrame.from_dict({'x_data':np.linspace(0,10,200)})
        df['y_data']=fun(df['x_data'],1,5,-0.2)
        df['noise']=np.random.normal(scale=0.2,size=200)
        df['y_sim']=df['y_data']+df['noise']
        
        arg_list = brute_leastsquare_fit(fun,df['x_data'],df['y_sim'],p_names=['a','b','c'],
                                         p_min_max_steps_dict={'a':[0,2,20],'b':[0,10,20],'c':[-1,1,20]},
                                         visualize=True)
        df['fitted_function']=fun(df['x_data'],*arg_list)
        df.plot('x_data')
    
    test_stitchSpectra=False
    plt.figure()
    if test_stitchSpectra:
        lamb_list=[]
        counts_list=[]    
        for i in range(4):
            lamb_list.append(np.linspace(500+i*100,700+i*100,256))
            counts_list.append(100*np.sin(lamb_list[-1]/100)**2+30*np.random.rand(256))
            plt.plot(lamb_list[-1],counts_list[-1])
            
        
        lamb,counts,coefficients=stitchSpectra(lamb_list, counts_list)
        plt.plot(lamb,counts,color='black')
        plt.xlabel('stitched lamb')
        plt.ylabel('stitched spectra')
        plt.show()
        
        
   
    
    