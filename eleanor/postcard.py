import os, sys

from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from lightkurve import search
import numpy as np
import warnings
import pandas as pd
import copy
from .mast import crossmatch_by_position

__all__ = ['Postcard']

class Postcard(object):
    """TESS FFI data for one postcard across one sector.
    
    A postcard is an rectangular subsection cut out from the FFIs. 
    It's like a TPF, but bigger. 
    The Postcard object contains a stack of these cutouts from all available 
    FFIs during a given sector of TESS observations.
    
    Parameters
    ----------
    filename : str
        Filename of the downloaded postcard.
    location : str, optional
        Filepath to `filename`.
    
    Attributes
    ----------
    dimensions : tuple
        (`x`, `y`, `time`) dimensions of postcard.
    flux, flux_err : numpy.ndarray
        Arrays of shape `postcard.dimensions` containing flux or error on flux 
        for each pixel.
    time : float
        ?
    header : dict
        Stored header information for postcard file.
    center_radec : tuple
        RA & Dec coordinates of the postcard's central pixel.
    center_xy : tuple
        (`x`, `y`) coordinates corresponding to the location of 
        the postcard's central pixel on the FFI.
    origin_xy : tuple
        (`x`, `y`) coordinates corresponding to the location of 
        the postcard's (0,0) pixel on the FFI.
    """
    def __init__(self, source, location=None):

        if location is not None:
            self.post_dir = location
        else:
            self.post_dir = os.path.join(os.path.expanduser('~'), '.eleanor', 'postcards')
            if os.path.isdir(self.post_dir) == False:
                try:
                    os.mkdir(self.post_dir)
                except OSError:
                    self.post_dir = '.'
                    warnings.warn('Warning: unable to create {}. '
                                  'Downloading postcard to the current '
                                  'working directory instead.'.format(self.post_dir))
            
        lk_post = search.search_tesscut('{0} {1}'.format(source.coords[0], source.coords[1])).download(cutout_size=133,
                                                                                          download_dir=self.post_dir)
        self.post_lk_obj = lk_post
        self.local_path = self.post_lk_obj.path


    def __repr__(self):
        return "eleanor postcard ({})".format(self.local_path)

    def plot(self, frame=0, ax=None, scale='linear', **kwargs):
        """Plots a single frame of a postcard.
        
        Parameters
        ----------
        frame : int, optional
            Index of frame. Default 0.
        
        ax : matplotlib.axes.Axes, optional
            Axes on which to plot. Creates a new object by default.
        
        scale : str
            Scaling for colorbar; acceptable inputs are 'linear' or 'log'.
            Default 'linear'.
        
        **kwargs : passed to matplotlib.pyplot.imshow
        
        Returns
        -------
        ax : matplotlib.axes.Axes
        """

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 7))
        if scale is 'log':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                dat = np.log10(self.flux[frame])
                dat[~np.isfinite(dat)] = np.nan
        else:
            dat = self.flux[frame]

        if ('vmin' not in kwargs) & ('vmax' not in kwargs):
            kwargs['vmin'] = np.nanpercentile(dat, 1)
            kwargs['vmax'] = np.nanpercentile(dat, 99)

        im = ax.imshow(dat, **kwargs)
        ax.set_xlabel('Row')
        ax.set_ylabel('Column')
        cbar = plt.colorbar(im, ax=ax)
        if scale == 'log':
            cbar.set_label('log$_{10}$ Flux')
        else:
            cbar.set_label('Flux')

        # Reset the x/y ticks to the position in the ACTUAL FFI.
#        xticks = ax.get_xticks() + self.center_xy[0]
#        yticks = ax.get_yticks() + self.center_xy[1]
#        ax.set_xticklabels(xticks)
#        ax.set_yticklabels(yticks)
        return ax

    def find_sources(self, radius=0.5):
        """Finds the cataloged sources in the postcard and returns a table.

        Returns
        -------
        result : astropy.table.Table
            All the sources in a postcard with TIC IDs or Gaia IDs.
        """
        result = crossmatch_by_position(self.center_radec, radius, 'Mast.Tic.Crossmatch').to_pandas()
        result = result[['MatchID', 'MatchRA', 'MatchDEC', 'pmRA', 'pmDEC', 'Tmag']]
        result.columns = ['TessID', 'RA', 'Dec', 'pmRA', 'pmDEC', 'Tmag']
        return result


    @property
    def header(self):
        return self.post_lk_obj.hdu[2].header

    @property
    def center_radec(self):
        return (self.header['CRVAL1'], self.header['CRVAL2'])

    @property
    def center_xy(self):
        return (self.header['CRPIX1'],  self.header['CRPIX2'])

    @property
    def flux(self):
        return self.post_lk_obj.flux

    @property
    def dimensions(self):
        return self.post_lk_obj.shape

    @property
    def flux_err(self):
        return self.post_lk_obj.flux_err
    
    @property
    def time(self):
        return self.post_lk_obj.time

    @property
    def wcs(self):
        return self.post_lk_obj.wcs

    ## LIGHTKURVE OBJECT HAS NO QUALITY FLAGS
    @property
    def quality(self):
        return self.hdu[1].data['QUALITY']

    ## LIGHTKURVE OBJECT HAS NO BACKGROUND
    @property
    def bkg(self):
        return self.hdu[1].data['BKG']
    
    ## LIGHTKURVE OBJECT HAS NO BARYCORR
    @property 
    def barycorr(self):
        return self.hdu[1].data['BARYCORR']
