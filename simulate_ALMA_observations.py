import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import casatools as cto
import casatasks as cta
from pathlib import Path
from typing import Union, List
from numpy.typing import NDArray

###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###
def run_simalma(project:str='sim',
                dryrun:bool=False,
                skymodel:str='',
                inbright:str='',
                indirection:str='',
                incell:str='',
                incenter:str='',
                inwidth:str='',
                #complist:str='',
                #compwidth:str='"0GHz"',
                #setpointings:bool=True,
                #ptgfile:str='$project.ptg.txt',
                #integration:str='0s',
                #direction:str='',
                mapsize:Union[str, List[str]]=" ",
                antennalist:List[str]=['alma.cycle1.1.cfg'],
                #hourangle:str='transit',
                totaltime:List[str]=['20min'],
                tpnant:int=0,
                tptime:str='0s',
                pwv:float=0.5,
                image:bool=False,
                #imsize:List[int]=[128, 128],
                #imdirection:str='',
                #cell:str='',
                #niter:int=0,
                #threshold:str='0.1mJy',
                #graphics:str='both',
                #verbose:bool=False,
                overwrite:bool=True,
                **kwargs):
    """
    This function runs the CASA task `simalma` to simulate ALMA observations. The `simalma` DocString is included below for reference.

    NOTE: **kwargs: Additional keyword arguments will be passed as such directly to simalma.

    .. _simalma:
    Simulation task for ALMA

    Parameters
    - project_ (string='sim') - root prefix for output file names
    - dryrun_ (bool=True) - dryrun=True will only produce the informative report, not run simobserve/analyze
    - skymodel_ (string='') - model image to observe

        .. raw:: html

            <details><summary><i> skymodel != '' </i></summary>

        - inbright_ (string='') - scale surface brightness of brightest pixel e.g. "1.2Jy/pixel"
        - indirection_ (string='') - set new direction e.g. "J2000 19h00m00 -40d00m00"
        - incell_ (string='') - set new cell/pixel size e.g. "0.1arcsec"
        - incenter_ (string='') - set new frequency of center channel e.g. "89GHz" (required even for 2D model)
        - inwidth_ (string='') - set new channel width e.g. "10MHz" (required even for 2D model)

        .. raw:: html

            </details>
    - complist_ (string='') - componentlist to observe

        .. raw:: html

            <details><summary><i> complist != '' </i></summary>

        - compwidth_ (string='"8GHz"') - bandwidth of components

        .. raw:: html

            </details>
    - setpointings_ (bool=True) - If True, calculate a map of pointings and write *ptgfile*.

        .. raw:: html

            <details><summary><i> setpointings = True </i></summary>

        - integration_ (string='10s') - integration (sampling) time
        - direction_ (stringVec='') - "J2000 19h00m00 -40d00m00" or "" to center on model
        - mapsize_ (stringVec=['', '']) - angular size of map or "" to cover model

        .. raw:: html

            </details>

        .. raw:: html

            <details><summary><i> setpointings = False </i></summary>

        - ptgfile_ (string='$project.ptg.txt') - list of pointing positions
        - integration_ (string='10s') - integration (sampling) time

        .. raw:: html

            </details>
    - antennalist_ (stringVec=['alma.cycle1.1.cfg', 'aca.cycle1.cfg']) - antenna position files of ALMA 12m and 7m arrays
    - hourangle_ (string='transit') - hour angle of observation center e.g. -3:00:00, or "transit"
    - totaltime_ (stringVec=['20min', '1h']) - total time of observation; vector corresponding to antennalist
    - tpnant_ (int=0) - Number of total power antennas to use (0-4)

        .. raw:: html

            <details><summary><i> tpnant != 0 </i></summary>

        - tptime_ (string='0s') - total observation time for total power

        .. raw:: html

            </details>
    - pwv_ (double=0.5) - Precipitable Water Vapor in mm. 0 for noise-free simulation
    - image_ (bool=True) - image simulated data

        .. raw:: html

            <details><summary><i> image = True </i></summary>

        - imsize_ (intVec=[128, 128]) - output image size in pixels (x,y) or 0 to match model
        - imdirection_ (string='') - set output image direction, (otherwise center on the model)
        - cell_ (string='') - cell size with units or "" to equal model
        - niter_ (int=0) - maximum number of iterations (0 for dirty image)
        - threshold_ (string='0.1mJy') - flux level (+units) to stop cleaning

        .. raw:: html

            </details>
    - graphics_ (string='both') - display graphics at each stage to [screen|file|both|none]
    - verbose_ (bool=False) - Print extra information to the logger and terminal.
    - overwrite_ (bool=False) - overwrite files starting with $project

    .. _Description:

    Description
    This task simulates ALMA observation including 12-m, ACA 7-m and
    total power arrays, and images and analyzes simulated data.

    This task makes multiple calls to **simobserve** (to calculate
    visibilities and total power spectra), followed by gridding of
    total power spectra (if total power is requested), concatenation
    of the simulated visibilities, calls to the **simanalyze** task
    for visibility inversion, deconvolution, and calculation of
    difference, and fidelity images, and feathering of single dish and
    interferometric data.

    These steps may not all be familiar to new users, so the
    **simalma** task runs by default in a "dryrun" mode, in which it
    assesses the user's input parameters and sky model, and prints an
    informational report including the required calls to other CASA
    tasks, both to the screen and to a text file in the project
    directory (defined below).

    The user can modify their parameters based on the information,
    then either run with *dryrun=False* to actually call the other
    tasks to create the simulated data, or run the other tasks
    individually one at a time to better understand and control the
    process. More information on running **simalma** can be found
    `here <../../notebooks/simulation.ipynb#ALMA-simulations>`__.

    .. note:: **NOTE**: The ALMA project is refining the optimal method of
        combining the three types of data. If that best practice is
        changed after this release of CASA, the user can control the
        process by modifying the calls to the other CASA tasks.


    .. warning:: **WARNING**: The simalma/simanalyze tasks do not support
        wideband multi-term imaging. Please use tclean (with other image
        analysis tasks) if your simulated MS from simobserve requires
        wideband continuum image reconstruction.

    .. rubric:: Parameter descriptions

    *project*

    The root filename for all output files. A subdirectory will be
    created, and all created files will be placed in that subdirectory
    including the informational report.

    *skymodel*

    An input image used as a model of the sky. **simalma** requires a
    CASA or FITS image. If you merely have a grid of numbers, you will
    need to write them out as FITS or write a CASA script to read them
    in and use the **image** (**ia**) tool to create an image and
    insert the data. **simalma** does NOT require a coordinate system
    in the header. If the coordinate information is incomplete,
    missing, or you would like to override it, set the appropriate
    "in" parameters.

    .. note:: **NOTE**: Setting those parameters simply changes the header
        values, ignoring any values already in the image. No regridding
        is performed.

    If you have a proper Coordinate System, **simalma** will do its
    best to generate visibilities from that, and then create a
    synthesis image
    according to the specified user parameters. You can manipulate
    an image header manually with the **imhead** task.

    *inbright*

    The peak brightness to scale the image to in Jy/pixel, or " " to
    keep it unchanged.

    .. note:: **NOTE**: "unchanged" will take the numerical values in your
        image and assume they are in Jy/pixel, even if it says some
        other unit in the header.

    *indirection*

    The central direction to place the sky model image, or " " to use
    whatever is in the image already.

    *incell*

    The spatial pixel size to scale the skymodel image, or " " to use
    whatever is in the image already.

    *incenter*

    The frequency to use for the center channel (or only channel, if
    the skymodel is 2D). E.g., "89GHz", or " " to use what is in the
    header. Required even for a 2D model.

    *inwidth*

    The width of channels to use, or " " to use what is in the image.
    Should be a string representing a quantity with units e.g.,
    "10MHz".

    .. note:: **NOTE**: Only works reliably with frequencies, not velocities.
        It is not possible to change the number of spectral planes of
        the sky model, only to relabel them with different frequencies.
        That kind of regridding can be accomplished with the CASA
        toolkit.

    *complist*

    A component list model of the sky, added to or instead of
    skymodel. Click
    `here <https://casaguides.nrao.edu/index.php/Simulation_Guide_Component_Lists_(CASA_5.1)>`__
    for more information.

    *compwidth*

    The bandwidth of components. If simulating from components only,
    this defines the bandwidth of the MS and output images.

    *setpointings*

    If True, calculate a map of pointings and write *ptgfile*. If
    graphics are on, display the pointings shown on the model image.
    Observations with the ALMA 12m and ACA 7m arrays will observe a
    region of size "mapsize" using the same hexagonal algorithm as the
    ALMA OT, with Nyquist sampling. The total power array maps a
    slightly (+1 primary beam) larger area than the 12m array does, to
    improve later image combination. It samples the region with
    lattice grids of spacing 0.33 lambda/D. If *setpointings=False*,
    read pointings from ptgfile.

    *ptgfile*

    A text file specifying directions in the same format as the
    example, and optional integration times, e.g.,

    ::

        #Epoch     RA          DEC      TIME(optional)
        J2000 23h59m28.10 -019d52m12.35 10.0

    If the time column is not present in the file, it will use
    "integration" for all pointings.

    .. note:: **NOTE**: At this time the file should contain only science
        pointings. **simalma** will observe these until totaltime is
        used up.

    *integration*

    Time interval for each integration e.g., '10s'.

    .. note:: **NOTE**: To simulate a "scan" longer than one integration, use
        *setpointings* to generate a pointing file, and then edit the
        file to increase the time at each point to be larger than the
        parameter integration time.

    *direction*

    Mosaic center direction. e.g., 'J2000 19h00m00 -40d00m00'. If
    unset, will use the center of the skymodel image. Can optionally
    be a list of pointings, otherwise **simobserve** will cover a
    region of size *mapsize* according to *maptype*.

    *mapsize*

    Angular size of mosaic map to simulate. Set to " " to cover the
    model image.

    *antennalist*

    A vector of ASCII files containing antenna positions, one for each
    configuration of 7m or 12m dishes.In this task, it should be an
    ALMA configuration. Standard arrays are found in your CASA data
    repository, os.getenv("CASAPATH").split()[0]+"/data/alma/simmos/".
    A string of the form "alma;0.5arcsec" will be parsed into a 12m
    ALMA configuration. Examples:
    ['alma.cycle2.5.cfg','aca.cycle2.i.cfg'],
    ['alma.cycle1;0.3arcsec','alma.cycle1.1.cfg','aca.i.cfg']

    *hourangle*

    Hour angle of observation e.g., '-3h'.

    *totaltime*

    The total time of observations. This should either be a scalar
    time quantity expressed as a string e.g., '1h', '3600sec',
    '10min', or a vector of such quantities, corresponding to the
    elements of the antennalist vector, e.g., ['5min','20min','3h'].
    If you specify a scalar, that will be used for the highest
    resolution 12m configuration in antennalist, and any lower
    resolution 12m configurations, any 7m configurations, and any TP
    configurations will have observing times relative to totaltime of
    0.5, 2,and 4, respectively.

    *tpnant*

    The number of total power antennas to use in simulation.

    *tptime*

    If *tpnant>0*, the user must specify the observing time for total
    power as a CASA quantity e.g., '4h'.

    .. note:: **NOTE**: This is not broken up among multiple days - a 20h
        track will include observations below the horizon,  which is
        probably not what is desired.

    *pwv*

    Precipitable water vapor. If constructing an atmospheric model,
    set 0 for noise-free simulation. When *pwv*>0, thermal noise is
    applied to the simulated data. J. Pardo's ATM library will be used
    to construct anatmospheric profile for the ALMA site: altitude
    5000m, ground pressure 650mbar, relhum=20%, a water layer of pwv
    at altitude of 2km, the sky brightness temperature returned by
    ATM, and internally tabulated receiver temperatures. See the
    documentation of **simobserve** for more details.

    *image*

    An option to invert and deconvolve the simulated MeasurementSet(s)

    .. note:: **NOTE**: Interactive clean or more parameters than the subset
        visible here are available by simply running either **clean**
        or **tclean** tasks directly.

    If graphics turned on, display the clean image and residual image
    uses Cotton-Schwab clean for single fields and Mosaic gridding for
    multiple fields (with Clark PSF calculation in minor cycles).

    *imsize*

    The image size in spatial pixels (x,y). 0 or -1 will use the model
    image size. Examples: imsize=[500,500]

    *imdirection*

    The phase center for synthesized image. Default is to center on
    the sky model.

    *cell*

    Cell size e.g., "10arcsec". *cell = " "* defaults to the skymodel
    cell.

    *niter*

    The number of clean/deconvolution iterations, 0 for no cleaning.

    *threshold*

    The flux level at which to stop cleaning.

    *graphics*

    View plots on the screen, saved to file, both, or neither.

    *verbose*

    Print extra information to the logger and terminal.

    *overwrite*

    Overwrite existing files in the project subdirectory. Please see
    the documents of **simobserve** and **simanalyze** for the list of
    outputs produced.

    .. _Examples:

    Examples
    Example of a **simalma** routine. More information on this can be
    seen
    `here <https://casaguides.nrao.edu/index.php/Simalma_(CASA_5.1)>`__.

    .. _Details:

    Parameter Details
    Detailed descriptions of each function parameter

    .. _project:

    | ``project (string='sim')`` - root prefix for output file names

    .. _dryrun:

    | ``dryrun (bool=True)`` - dryrun=True will only produce the informative report, not run simobserve/analyze

    .. _skymodel:

    | ``skymodel (string='')`` - model image to observe

    .. _inbright:

    | ``inbright (string='')`` - scale surface brightness of brightest pixel e.g. "1.2Jy/pixel"

    .. _indirection:

    | ``indirection (string='')`` - set new direction e.g. "J2000 19h00m00 -40d00m00"

    .. _incell:

    | ``incell (string='')`` - set new cell/pixel size e.g. "0.1arcsec"

    .. _incenter:

    | ``incenter (string='')`` - set new frequency of center channel e.g. "89GHz" (required even for 2D model)

    .. _inwidth:

    | ``inwidth (string='')`` - set new channel width e.g. "10MHz" (required even for 2D model)

    .. _complist:

    | ``complist (string='')`` - componentlist to observe

    .. _compwidth:

    | ``compwidth (string='"8GHz"')`` - bandwidth of components

    .. _setpointings:

    | ``setpointings (bool=True)`` - If True, calculate a map of pointings and write *ptgfile*.

    .. _ptgfile:

    | ``ptgfile (string='$project.ptg.txt')`` - list of pointing positions

    .. _integration:

    | ``integration (string='10s')`` - integration (sampling) time

    .. _direction:

    | ``direction (stringVec='')`` - "J2000 19h00m00 -40d00m00" or "" to center on model

    .. _mapsize:

    | ``mapsize (stringVec=['', ''])`` - angular size of map or "" to cover model

    .. _antennalist:

    | ``antennalist (stringVec=['alma.cycle1.1.cfg', 'aca.cycle1.cfg'])`` - antenna position files of ALMA 12m and 7m arrays

    .. _hourangle:

    | ``hourangle (string='transit')`` - hour angle of observation center e.g. -3:00:00, or "transit"

    .. _totaltime:

    | ``totaltime (stringVec=['20min', '1h'])`` - total time of observation; vector corresponding to antennalist

    .. _tpnant:

    | ``tpnant (int=0)`` - Number of total power antennas to use (0-4)

    .. _tptime:

    | ``tptime (string='0s')`` - total observation time for total power

    .. _pwv:

    | ``pwv (double=0.5)`` - Precipitable Water Vapor in mm. 0 for noise-free simulation

    .. _image:

    | ``image (bool=True)`` - image simulated data

    .. _imsize:

    | ``imsize (intVec=[128, 128])`` - output image size in pixels (x,y) or 0 to match model

    .. _imdirection:

    | ``imdirection (string='')`` - set output image direction, (otherwise center on the model)

    .. _cell:

    | ``cell (string='')`` - cell size with units or "" to equal model

    .. _niter:

    | ``niter (int=0)`` - maximum number of iterations (0 for dirty image)

    .. _threshold:

    | ``threshold (string='0.1mJy')`` - flux level (+units) to stop cleaning

    .. _graphics:

    | ``graphics (string='both')`` - display graphics at each stage to [screen|file|both|none]

    .. _verbose:

    | ``verbose (bool=False)`` - Print extra information to the logger and terminal.

    .. _overwrite:

    | ``overwrite (bool=False)`` - overwrite files starting with $project
    """

    params = {
        "project": project,
        "dryrun": dryrun,
        "skymodel": skymodel,
        "inbright": inbright,
        "indirection": indirection,
        "incell": incell,
        "incenter": incenter,
        "inwidth": inwidth,
        # "complist": complist,
        # "compwidth": compwidth,
        # "setpointings": setpointings,
        # "ptgfile": ptgfile,
        # "integration": integration,
        # "direction": direction,
        "mapsize": mapsize,
        "antennalist": antennalist,
        # "hourangle": hourangle,
        "totaltime": totaltime,
        "tpnant": tpnant,
        "tptime": tptime,
        "pwv": pwv,
        "image": image,
        # "imsize": imsize,
        # "imdirection": imdirection,
        # "cell": cell,
        # "niter": niter,
        # "threshold": threshold,
        # "graphics": graphics,
        # "verbose": verbose,
        "overwrite": overwrite
    }

    print("Running simalma with the following parameters:")
    for key, value in params.items():
        if value not in [False, None, "", [],[""]]:
            print(f"{key}: {value}")

    cta.simalma(
        project=project,
        dryrun=dryrun,
        skymodel=skymodel,
            inbright=inbright,
            indirection=indirection,
            incell=incell,
            incenter=incenter,
            inwidth=inwidth,
        # complist=complist,
        #     compwidth=compwidth,
        # setpointings=setpointings,
        #     ptgfile=ptgfile,
        #     integration=integration,
        #     direction=direction,
            mapsize=mapsize,
        antennalist=antennalist,
        # hourangle=hourangle,
        totaltime=totaltime,
        tpnant=tpnant,
            tptime=tptime,
        pwv=pwv,
        image=image,
        #     imsize=imsize,
        #     imdirection=imdirection,
        #     cell=cell,
        #     niter=niter,
        #     threshold=threshold,
        # graphics=graphics,
        # verbose=verbose,
        overwrite=overwrite,
        **kwargs)
###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###

###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###
def extract_MS_data(ms_path:str, npz_file:str, make_visibility_plots:bool=True):
    """
    Extracts data from the simulated ALMA Measurement Set (MS) and returns it as a .npz file, for use in MPol RML imaging.

    Parameters:
        - ms_path (str): Path to the simulated ALMA Measurement Set (MS) file.
        - npz_file (str): Path where the extracted data will be saved as a .npz file.
        - make_visibility_plots (bool): If True, generates visibility plots from the extracted data.

    Raises:
        - FileNotFoundError: If the MS file does not exist.
        - RuntimeError: If the MS file is empty or has no data.

    Returns:
        - n_chan (List[int]): Number of channels per spectral window in the MS.
        - n_corr (List[int]): Number of correlations per spectral window in the MS.
        - u (NDArray[float]): U coordinates of the visibility data.
        - v (NDArray[float]): V coordinates of the visibility data.
        - data (NDArray[complex]): Visibility data extracted from the MS.
        - amp (NDArray[float]): Amplitude of the visibility data.
        - phase (NDArray[float]): Phase of the visibility data.
        - real (NDArray[float]): Real part of the visibility data.
        - imag (NDArray[float]): Imaginary part of the visibility data.
        - uvdist (NDArray[float]): UV distance of the visibility data.
        - weight (NDArray[float]): Weights of the visibility data.
        - sigma (NDArray[float]): Sigma values of the visibility data.
        - data_desc_id (NDArray[int]): Data description IDs from the MS.
    """

    ms = cto.ms()

    msp = Path(ms_path)
    if not msp.exists():
        raise FileNotFoundError(f"The Measurement Set file {ms_path} does not exist.")

    ms.open(ms_path)

    spw_info = ms.getspectralwindowinfo()
    print("Number of SPWs in MS:", len(spw_info))
    n_chan = [spw_info[str(i)]['NumChan'] for i in range(len(spw_info))]
    print("Channels per SPW:", n_chan)
    n_corr = [spw_info[str(i)]['NumCorr'] for i in range(len(spw_info))]
    print("Corrs./Pols. per SPW:", n_corr, '\n')

    ms_data = ms.getdata(['u', 'v', 'data', 'amplitude', 'phase', 'real', 'imaginary', 'uvdist', 'weight', 'sigma', 'data_desc_id'])
    u = ms_data['u']
    v = ms_data['v']
    data = ms_data['data'] # (n_corr, n_chan, n_rows)
    amp = ms_data['amplitude']
    phase = ms_data['phase']
    real = ms_data['real']
    imag = ms_data['imaginary']
    uvdist = ms_data['uvdist']
    weight = ms_data['weight']
    sigma = ms_data['sigma']
    data_desc_id = ms_data['data_desc_id']

    data  = data.mean(axis=0).squeeze()
    amp = amp.mean(axis=0).squeeze()
    phase = phase.mean(axis=0).squeeze()
    real = real.mean(axis=0).squeeze()
    imag = imag.mean(axis=0).squeeze()
    weight = weight.mean(axis=0)
    sigma = sigma.mean(axis=0)

    print("Data shape:", data.shape)

    ms.close()

    if make_visibility_plots:
        plot_visibilities(u=u, v=v, uvdist=uvdist, amp=amp, phase=phase, real=real, imag=imag)

    np.savez(npz_file, u=u, v=v, data=data, weight=weight)

    return n_chan, n_corr, u, v, data, amp, phase, real, imag, uvdist, weight, sigma, data_desc_id
###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###

###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###
def plot_visibilities(u:NDArray[float], v:NDArray[float], uvdist:NDArray[float], amp:NDArray[float], phase:NDArray[float], real:NDArray[float], imag:NDArray[float]):
    """
    Plots the visibility data extracted from the ALMA Measurement Set. The plots are saved as PDF files.

    Parameters:
        u (NDArray[float]): U coordinates of the visibility data.
        v (NDArray[float]): V coordinates of the visibility data.
        uvdist (NDArray[float]): UV distance of the visibility data.
        amp (NDArray[float]): Amplitude of the visibility data.
        phase (NDArray[float]): Phase of the visibility data.
        real (NDArray[float]): Real part of the visibility data.
        imag (NDArray[float]): Imaginary part of the visibility data.

    Raises:
        ValueError: If the input arrays are empty or have mismatched dimensions.

    Returns:
        None
    """

    if not all(arr.size > 0 for arr in [u, v, uvdist, amp, phase, real, imag]):
        raise ValueError("One or more input arrays are empty.")
    if not (u.shape == v.shape == uvdist.shape == amp.shape == phase.shape == real.shape == imag.shape):
        print(f"Shapes of input arrays:\nu: {u.shape}, v: {v.shape}, uvdist: {uvdist.shape}, amp: {amp.shape}, phase: {phase.shape}, real: {real.shape}, imag: {imag.shape}")
        raise ValueError("Input arrays must have the same shape.")

    fig, ax = plt.subplots()
    ax.scatter(u, v, s=1.5, rasterized=True, linewidths=0.0, c="k")
    ax.scatter(
        -u, -v, s=1.5, rasterized=True, linewidths=0.0, c="k"
    )
    ax.set_xlabel(r"$u$ [k$\lambda$]")
    ax.set_ylabel(r"$v$ [k$\lambda$]")
    ax.set_title("uv-coverage of ALMA simulated data")
    plt.savefig("uv_coverage.pdf", format='pdf', bbox_inches="tight")

    fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(8, 8))
    pkw = {"s":1, "rasterized":True, "linewidths":0.0, "c":"k"}
    ax[0].scatter(uvdist, amp, **pkw)
    ax[0].set_ylabel("amplitude [Jy]")
    ax[1].scatter(uvdist, phase, **pkw)
    ax[1].set_ylabel("phase [radians]")
    ax[2].scatter(uvdist, real, **pkw)
    ax[2].set_ylabel("Re(V) [Jy]")
    ax[3].scatter(uvdist, imag, **pkw)
    ax[3].set_ylabel("Im(V) [Jy]")
    ax[3].set_xlabel(r"$uv$dist [k$\lambda$]")
    plt.savefig("visibility_plots.pdf", format='pdf', bbox_inches="tight")
###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###