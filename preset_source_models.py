import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from typing import Union, List, Tuple

class ModelSourceMaker:
    """
    A class to create and manipulate model source images that can serve as inputs for ALMA simulations. This class provides methods to generate various source models such as flat disks, Gaussian profiles, elliptical disks, hollow rings, and concentric rings. It also includes methods for plotting the model images and saving them in FITS format.
    The models are defined on a square grid with a specified field of view (fov) and pixel scale. The center of the image can be specified in RA and Dec coordinates.

    Attributes:
        - fov_arcsec (float): Field of view in arcseconds (default: 1 arcsec).
        - npix (int): Number of pixels along one dimension of the image (default: 128).
        - pixel_scale_arcsec (float): Pixel scale in arcseconds.
        - pixel_scale_deg (float): Pixel scale in degrees.
        - shape (tuple): Shape of the image array in pixel units.
        - ra_dec_center: Center coordinates of the image in RA (h.m.s) and DEC (d.m.s) (default: "00h00m00.0s 00d00m00.0s").
        - xx, yy: Meshgrid arrays for pixel coordinates in the image.
        - rr: Radial distance from the center in arcseconds.

    Methods:
        - _empty_image(): Returns an empty image array of the specified shape.
        - flat_disk(radius_arcsec, intensity=1.0): Generates a flat disk model.
        - gaussian(fwhm_arcsec, amplitude=1.0): Generates a Gaussian model.
        - elliptical_disk(major_arcsec, minor_arcsec, angle_deg, intensity=1.0): Generates an elliptical disk model.
        - hollow_ring(inner_radius_arcsec, outer_radius_arcsec, intensity=1.0): Generates a hollow ring model.
        - concentric_rings(radii_arcsec, width_arcsec, intensity=1.0): Generates concentric rings model.
        - plot_image(image, model_name="Model", cmap="hot", save_pdf=True): Plots the image and saves it as a PDF.
        - save_fits(image, filename, bunit="Jy/pixel", object_name="Model Source"): Saves the image in FITS format.

    Example usage:
        >>> maker = ModelSourceMaker(fov_arcsec=2, npix=256, ra_dec_center="12h30m00.0s -30d00m00.0s")
        >>> disk_image = maker.flat_disk(radius_arcsec=0.5, intensity=10.0)
        >>> maker.plot_image(disk_image, model_name="Flat Disk Model")
        >>> maker.save_fits(disk_image, filename="flat_disk_model.fits", object_name="Flat Disk Source")
    """

    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###
    def __init__(self, fov_arcsec:float=1, npix:int=128, ra_dec_center:str="00h00m00.0s 00d00m00.0s"):
        """
        Initializes the ModelSourceMaker with a specified field of view, number of pixels, and center coordinates. This sets up the pixel scale, shape of the image, and creates meshgrid arrays for pixel coordinates.

        Parameters:
            fov_arcsec (float): Field of view in arcseconds (default: 1 arcsec).
            npix (int): Number of pixels along one dimension of the image (default: 128).
            ra_dec_center (str or SkyCoord): Center coordinates of the image in RA (h.m.s) and DEC (d.m.s) format or as a SkyCoord object (default: "00h00m00.0s 00d00m00.0s").

        Raises:
            ValueError: If fov_arcsec or npix are not positive numbers.
            TypeError: If ra_dec_center is not a valid SkyCoord object or string.

        Returns:
            None
        """

        if not isinstance(fov_arcsec, (int, float)) or fov_arcsec <= 0:
            raise ValueError("Field of view must be a positive number in arcseconds.")
        if not isinstance(npix, int) or npix <= 0:
            raise ValueError("Number of pixels must be a positive integer.")
        if isinstance(ra_dec_center, str):
            try:
                self.ra_dec_center = SkyCoord(ra_dec_center, unit=(u.hourangle, u.deg))
            except Exception as e:
                raise TypeError(f"Invalid RA/Dec center format: {e}")

        self.fov_arcsec = fov_arcsec
        self.npix = npix
        self.pixel_scale_arcsec = fov_arcsec / npix
        self.pixel_scale_deg = self.pixel_scale_arcsec / 3600.0
        self.shape = (npix, npix)
        self.ra_dec_center = SkyCoord(ra_dec_center, unit=(u.hourangle, u.deg))

        x = np.arange(npix) - npix // 2
        y = np.arange(npix) - npix // 2
        self.xx, self.yy = np.meshgrid(x, y)
        self.rr = np.sqrt(self.xx**2 + self.yy**2) * self.pixel_scale_arcsec
    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###

    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###
    def _empty_image(self):
        """
        Creates an empty image array of the specified shape initialized to zero. This method is used internally to create a blank image for various model generation methods.

        Parameters:
            None

        Raises:
            None

        Returns:
            numpy.ndarray: A 2D array of zeros with the shape defined by self.shape.
        """

        return np.zeros(self.shape)
    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###

    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###
    def flat_disk(self, radius_arcsec:float, intensity:float=1.0):
        """
        Generates a 2D flat disk model image. This method creates a circular disk with a specified radius and intensity. The disk is centered in the image, and all pixels within the radius are set to the specified intensity.

        Parameters:
            radius_arcsec (float): Radius of the disk in arcseconds.
            intensity (float): Intensity of the disk (default: 1.0).

        Raises:
            ValueError: If radius_arcsec is not a positive number.
            ValueError: If intensity is not a numeric value.

        Returns:
            numpy.ndarray: 2D array representing the flat disk model image.
        """

        if not isinstance(radius_arcsec, (int, float)) or radius_arcsec <= 0:
            raise ValueError("Radius must be a positive number in arcseconds.")
        if not isinstance(intensity, (int, float)):
            raise ValueError("Intensity must be a numeric value.")

        image = self._empty_image()
        image[self.rr <= radius_arcsec] = intensity
        return image
    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###

    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###
    def gaussian(self, fwhm_arcsec:float, amplitude:float=1.0):
        """
        Generates a 2D circular Gaussian model image. This method creates a Gaussian profile centered in the image with a specified Full Width at Half Maximum (FWHM) and peak amplitude. The Gaussian is computed based on the radial distance from the center of the image.

        Parameters:
            fwhm_arcsec (float): Full Width at Half Maximum of the Gaussian in arcseconds.
            amplitude (float): Peak intensity of the Gaussian (default: 1.0).

        Raises:
            ValueError: If fwhm_arcsec is not a positive number.
            ValueError: If amplitude is not a numeric value.

        Returns:
            numpy.ndarray: 2D array representing the Gaussian model image.
        """

        if not isinstance(fwhm_arcsec, (int, float)) or fwhm_arcsec <= 0:
            raise ValueError("FWHM must be a positive number in arcseconds.")
        if not isinstance(amplitude, (int, float)):
            raise ValueError("Amplitude must be a numeric value.")

        sigma_arcsec = fwhm_arcsec / (2.0 * np.sqrt(2 * np.log(2)))
        x_arcsec = self.xx * self.pixel_scale_arcsec
        y_arcsec = self.yy * self.pixel_scale_arcsec
        r2_arcsec = x_arcsec**2 + y_arcsec**2

        image = self._empty_image()
        image = amplitude * np.exp(-r2_arcsec / (2 * sigma_arcsec**2))
        return image
    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###

    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###
    def elliptical_disk(self, major_arcsec:float, minor_arcsec:float, angle_deg:float, intensity:float=1.0):
        """
        Generates a 2D elliptical disk model image. This method creates an elliptical disk with specified lengths of the major and minor axes, a position angle, and intensity. The ellipse is centered in the image, and all pixels within the ellipse are set to the specified intensity.

        Parameters:
            major_arcsec (float): Length of the major axis of the ellipse in arcseconds.
            minor_arcsec (float): Length of the minor axis of the ellipse in arcseconds.
            angle_deg (float): Position angle of the ellipse in degrees (measured from North due East).
            intensity (float): Intensity of the elliptical disk (default: 1.0).

        Raises:
            ValueError: If major_arcsec or minor_arcsec are not positive numbers.
            ValueError: If intensity is not a numeric value.
            ValueError: If angle_deg is not a numeric value.

        Returns:
            numpy.ndarray: 2D array representing the elliptical disk model image.
        """

        if not isinstance(major_arcsec, (int, float)) or major_arcsec <= 0:
            raise ValueError("Major axis length must be a positive number in arcseconds.")
        if not isinstance(minor_arcsec, (int, float)) or minor_arcsec <= 0:
            raise ValueError("Minor axis length must be a positive number in arcseconds.")
        if not isinstance(angle_deg, (int, float)):
            raise ValueError("Angle must be a numeric value in degrees.")
        if not isinstance(intensity, (int, float)):
            raise ValueError("Intensity must be a numeric value.")

        theta = (np.pi / 2) - np.deg2rad(angle_deg)
        x_rot = self.xx * np.cos(theta) + self.yy * np.sin(theta)
        y_rot = -self.xx * np.sin(theta) + self.yy * np.cos(theta)
        r_ell = (x_rot / (major_arcsec / self.pixel_scale_arcsec))**2 + (y_rot / (minor_arcsec / self.pixel_scale_arcsec))**2

        image = self._empty_image()
        image[r_ell <= 1.0] = intensity
        return image
    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###

    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###
    def hollow_ring(self, inner_radius_arcsec:float, outer_radius_arcsec:float, intensity:float=1.0):
        """
        Generates a 2D hollow ring model image. This method creates a ring with specified inner and outer radii, centered in the image. All pixels within the ring are set to the specified intensity.

        Parameters:
            inner_radius_arcsec (float): Inner radius of the ring in arcseconds.
            outer_radius_arcsec (float): Outer radius of the ring in arcseconds.
            intensity (float): Intensity of the ring (default: 1.0).

        Raises:
            ValueError: If inner_radius_arcsec or outer_radius_arcsec are not positive numbers.
            ValueError: If inner_radius_arcsec is greater than or equal to outer_radius_arcsec.
            ValueError: If intensity is not a numeric value.

        Returns:
            numpy.ndarray: 2D array representing the hollow ring model image.
        """

        if not isinstance(inner_radius_arcsec, (int, float)) or inner_radius_arcsec <= 0:
            raise ValueError("Inner radius must be a positive number in arcseconds.")
        if not isinstance(outer_radius_arcsec, (int, float)) or outer_radius_arcsec <= 0:
            raise ValueError("Outer radius must be a positive number in arcseconds.")
        if inner_radius_arcsec >= outer_radius_arcsec:
            raise ValueError("Inner radius must be less than outer radius.")
        if not isinstance(intensity, (int, float)):
            raise ValueError("Intensity must be a numeric value.")

        image = self._empty_image()
        mask = (self.rr >= inner_radius_arcsec) & (self.rr <= outer_radius_arcsec)
        image[mask] = intensity
        return image
    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###

    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###
    def concentric_rings(self, radii_arcsec:List[float], widths_arcsec:Union[float, List[float]], intensities:Union[float, List[float]]):
        """
        Generates a 2D image with concentric rings. This method creates a model image with multiple concentric rings, where each ring can have a different width and intensity. The rings are defined by their radii in arcseconds, and the image is centered around the specified field of view.

        Parameters:
            radii_arcsec (list[float]): List of radii for the concentric rings in arcseconds.
            widths_arcsec (float or list[float]): Width(s) of the rings in arcseconds. Can be a single float or a list matching radii_arcsec.
            intensities (float or list[float]): Intensity value(s) for the rings. Can be a single float or a list matching radii_arcsec.

        Raises:
            ValueError: If radii_arcsec is not a non-empty list or array.
            ValueError: If widths_arcsec is not a float or a list of floats of the same length as radii_arcsec.
            ValueError: If intensities is not a float or a list of floats of the same length as radii_arcsec.
            ValueError: If any ring width is not positive.

        Returns:
            numpy.ndarray: 2D array representing the concentric rings model image.
        """

        if not isinstance(radii_arcsec, (list, np.ndarray)) or len(radii_arcsec) == 0:
            raise ValueError("radii_arcsec must be a non-empty list or array of arcseconds.")

        if isinstance(widths_arcsec, (int, float)):
            widths = [widths_arcsec] * len(radii_arcsec)
        elif isinstance(widths_arcsec, (list, np.ndarray)) and len(widths_arcsec) == len(radii_arcsec):
            widths = widths_arcsec
        else:
            raise ValueError("widths_arcsec must be a float or a list of the same length as radii_arcsec.")

        if isinstance(intensities, (int, float)):
            intens = [intensities] * len(radii_arcsec)
        elif isinstance(intensities, (list, np.ndarray)) and len(intensities) == len(radii_arcsec):
            intens = intensities
        else:
            raise ValueError("intensities must be a float or a list of the same length as radii_arcsec.")

        if not all(isinstance(w, (int, float)) and w > 0 for w in widths):
            raise ValueError("All ring widths must be positive numbers.")
        if not all(isinstance(i, (int, float)) for i in intens):
            raise ValueError("All intensity values must be numeric.")

        image = self._empty_image()
        for r, w, i in zip(radii_arcsec, widths, intens):
            mask = (self.rr >= r - w / 2) & (self.rr <= r + w / 2)
            image[mask] = i
        return image
    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###

    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###
    def elliptical_gaussian(self, major_fwhm_arcsec:float, minor_fwhm_arcsec:float, angle_deg:float, amplitude:float=1.0):
        """
        Generates a 2D elliptical Gaussian model image. This method creates an elliptical Gaussian profile centered in the image with specified Full Width at Half Maximum (FWHM) for the major and minor axes, a position angle, and peak amplitude. The Gaussian is computed based on the radial distance from the center of the image, adjusted for the elliptical shape.

        Parameters:
            major_fwhm_arcsec (float): Full width at half maximum of the major axis in arcseconds.
            minor_fwhm_arcsec (float): Full width at half maximum of the minor axis in arcseconds.
            angle_deg (float): Position angle of the major axis in degrees (measured from North due East).
            amplitude (float): Peak amplitude of the Gaussian (default: 1.0).

        Raises:
            ValueError: If major_fwhm_arcsec or minor_fwhm_arcsec are not positive numbers.
            ValueError: If angle_deg is not a numeric value.
            ValueError: If amplitude is not a numeric value.

        Returns:
            numpy.ndarray: 2D array representing the elliptical Gaussian model image.
        """

        if not isinstance(major_fwhm_arcsec, (int, float)) or major_fwhm_arcsec <= 0:
            raise ValueError("Major FWHM must be a positive number in arcseconds.")
        if not isinstance(minor_fwhm_arcsec, (int, float)) or minor_fwhm_arcsec <= 0:
            raise ValueError("Minor FWHM must be a positive number in arcseconds.")
        if not isinstance(angle_deg, (int, float)):
            raise ValueError("Angle must be a numeric value in degrees.")
        if not isinstance(amplitude, (int, float)):
            raise ValueError("Amplitude must be a numeric value.")

        theta = (np.pi / 2) - np.deg2rad(angle_deg)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        x_rot = self.xx * cos_theta + self.yy * sin_theta
        y_rot = -self.xx * sin_theta + self.yy * cos_theta
        sigma_major = (major_fwhm_arcsec / self.pixel_scale_arcsec) / (2 * np.sqrt(2 * np.log(2)))
        sigma_minor = (minor_fwhm_arcsec / self.pixel_scale_arcsec) / (2 * np.sqrt(2 * np.log(2)))

        image = self._empty_image()
        image = amplitude * np.exp(-((x_rot / sigma_major)**2 + (y_rot / sigma_minor)**2) / 2)
        return image
    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###

    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###
    def spiral_arms(self, arm_width_arcsec:float, pitch_angle_deg:float, number_of_turns:int=2, intensity:float=1.0):
        """
        Generates a 2D model image with logarithmic spiral arms. This method creates a logarithmic spiral pattern with specified arm width, pitch angle, and number of turns. The spiral arms are centered in the image, and all pixels within the arms are set to the specified intensity.

        Parameters:
            arm_width_arcsec (float): Width of the spiral arm in arcseconds.
            pitch_angle_deg (float): Pitch angle of the spiral in degrees.
            number_of_turns (int): Number of spiral windings around the center (default: 2).
            intensity (float): Intensity of the spiral arms (default: 1.0).

        Raises:
            ValueError: If arm_width_arcsec is not a positive number.
            ValueError: If pitch_angle_deg is not a numeric value.
            ValueError: If number_of_turns is not a positive integer.
            ValueError: If intensity is not a numeric value.

        Returns:
            numpy.ndarray: 2D array representing the spiral arms model image.
        """

        if not isinstance(arm_width_arcsec, (int, float)) or arm_width_arcsec <= 0:
            raise ValueError("Arm width must be a positive number in arcseconds.")
        if not isinstance(pitch_angle_deg, (int, float)):
            raise ValueError("Pitch angle must be a numeric value in degrees.")
        if not isinstance(number_of_turns, int) or number_of_turns <= 0:
            raise ValueError("Number of turns must be a positive integer.")
        if not isinstance(intensity, (int, float)):
            raise ValueError("Intensity must be a numeric value.")

        theta = np.arctan2(self.yy, self.xx)
        r_arcsec = self.rr

        pitch_rad = np.deg2rad(pitch_angle_deg)
        with np.errstate(divide='ignore', invalid='ignore'):
            spiral_theta = np.full_like(r_arcsec, np.nan)
            valid_mask = r_arcsec > 0
            spiral_theta[valid_mask] = np.log(r_arcsec[valid_mask] / self.pixel_scale_arcsec) / np.tan(pitch_rad)

        distance_from_spiral = np.abs(((theta - spiral_theta) + np.pi) % (2 * np.pi) - np.pi)

        mask = (r_arcsec <= number_of_turns * 360 * self.pixel_scale_arcsec) & \
               (distance_from_spiral < (arm_width_arcsec / r_arcsec.clip(min=1e-6)))

        image = self._empty_image()
        image[mask] = intensity
        return image
    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###

    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###
    def point_sources(self, source_list:List[Tuple[float, float, float]]):
        """
        Generates a 2D image with multiple point sources (delta functions). This method creates an image with point sources defined by their RA and Dec offsets from the center of the image, along with their intensities. Each point source is represented as a delta function in the image.

        Parameters:
            source_list (list of tuples): Each tuple should be (RA_offset_arcsec, Dec_offset_arcsec, intensity). RA/Dec offsets are relative to image center in arcseconds.

        Raises:
            ValueError: If source_list is not a list or if any tuple does not have exactly three elements.
            TypeError: If RA/Dec offsets or intensity are not numeric values.

        Returns:
            numpy.ndarray: 2D array representing the point source model image.
        """

        if not isinstance(source_list, list) or not all(isinstance(src, tuple) and len(src) == 3 for src in source_list):
            raise ValueError("Source list must be a list of tuples with (RA_offset_arcsec, Dec_offset_arcsec, intensity).")
        for ra_offset, dec_offset, intensity in source_list:
            if not isinstance(ra_offset, (int, float)) or not isinstance(dec_offset, (int, float)) or not isinstance(intensity, (int, float)):
                raise TypeError("RA/Dec offsets and intensity must be numeric values.")

        image = self._empty_image()
        for ra_offset, dec_offset, intensity in source_list:
            x_pix = int(self.npix / 2 + ra_offset / self.pixel_scale_arcsec)
            y_pix = int(self.npix / 2 + dec_offset / self.pixel_scale_arcsec)
            if 0 <= x_pix < self.npix and 0 <= y_pix < self.npix:
                image[y_pix, x_pix] += intensity
        return image
    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###

    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###
    def plot_image(self, image:np.ndarray, model_name:str = "Model", cmap:str = "hot", bunit:str = "Jy/pixel", save_pdf:bool = True):
        """
        Plots the given image and saves it (optional) as a PDF file. This method creates a 2D plot of the model image with appropriate axes labels and a colorbar. The plot is centered around the specified field of view and uses the specified colormap.

        Parameters:
            image (numpy.ndarray): 2D array representing the model image to be plotted.
            model_name (str): Name of the model, used in the plot title and filename (default: "Model").
            cmap (str): Colormap to use for the image (default: "hot").
            bunit (str): Data unit for the colorbar label (default: "Jy/pixel").
            save_pdf (bool): Whether to save the plot as a PDF file (default: True).

        Raises:
            ValueError: If image is not a 2D numpy array.
            TypeError: If model_name is not a string or if cmap is not a valid colormap name.
            TypeError: If bunit is not a string.

        Returns:
            None
        """

        if not isinstance(image, np.ndarray) or image.ndim != 2:
            raise ValueError("Image must be a 2D numpy array.")
        if not isinstance(model_name, str):
            raise TypeError("Model name must be a string.")
        if isinstance(cmap, str):
            if cmap not in plt.colormaps():
                raise ValueError(f"Invalid colormap name: {cmap}. Please use a valid matplotlib colormap.")
        else:
            raise TypeError("Colormap must be a string representing a valid matplotlib colormap.")
        if not isinstance(bunit, str):
            raise TypeError("BUNIT must be a string.")

        extent = [-self.fov_arcsec, self.fov_arcsec, -self.fov_arcsec, self.fov_arcsec]

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(image, origin='lower', extent=extent, cmap=cmap)
        ax.set_xlabel("RA Offset [arcsec]")
        ax.set_ylabel("Dec Offset [arcsec]")
        ax.set_title(model_name)
        ax.invert_xaxis()

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        cb = plt.colorbar(im, cax=cax)
        cb.set_label(bunit)
        cb.set_ticks(np.linspace(np.min(image), np.max(image), 5))

        plt.tight_layout()

        if save_pdf:
            plt.savefig(f"{model_name}.pdf", bbox_inches="tight")
            plt.close()
        else:
            plt.show()
    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###

    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###
    def save_fits(self, image:np.ndarray, filename:str, bunit:str="Jy/pixel", object_name:str="Model Source"):
        """
        Saves the given image as a FITS file with the specified header information. This method creates a FITS file with WCS information based on the model's pixel scale and center coordinates. The image data is stored in the primary HDU, and the header includes the specified coordinates, data unit and object name.

        Parameters:
            image (numpy.ndarray): 2D array representing the model image to be saved.
            filename (str): Name of the output FITS file.
            bunit (str): Data unit for the FITS file (default: "Jy/pixel").
            object_name (str): Name of the object for the FITS header (default: "Model Source").

        Raises:
            ValueError: If image is not a 2D numpy array.
            TypeError: If filename is not a string or if bunit/object_name are not strings.
            TypeError: If the WCS cannot be created due to invalid parameters.

        Returns:
            None
        """

        if not isinstance(image, np.ndarray) or image.ndim != 2:
            raise ValueError("Image must be a 2D numpy array.")
        if not isinstance(filename, str):
            raise TypeError("Filename must be a string.")
        if not isinstance(bunit, str):
            raise TypeError("BUNIT must be a string.")
        if not isinstance(object_name, str):
            raise TypeError("Object name must be a string.")
        if image.shape != self.shape:
            raise ValueError(f"Image shape {image.shape} does not match expected shape {self.shape}.")

        w = WCS(naxis=2)
        w.wcs.crpix = [self.npix // 2, self.npix // 2]
        w.wcs.cdelt = np.array([-self.pixel_scale_deg, self.pixel_scale_deg])
        w.wcs.crval = [self.ra_dec_center.ra.deg, self.ra_dec_center.dec.deg]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.cunit = ["deg", "deg"]

        hdu = fits.PrimaryHDU(data=np.fliplr(image), header=w.to_header())
        hdu.header['BUNIT'] = bunit
        hdu.header['OBJECT'] = object_name
        hdu.writeto(filename, overwrite=True)
    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###

    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###
    def generate_and_save(self, model_name:str, image:np.ndarray, cmap:str = "hot", bunit:str="Jy/pixel", save_fits:bool = True, save_pdf:bool = True):
        """
        Helper function to save the model image as a FITS file and plot (and optionally save) as PDF.

        Parameters:
            model_name (str): Name for the model, used for filenames and plot titles.
            image (np.ndarray): 2D array representing the model image.
            save_fits (bool): Whether to save the FITS file (default: True).
            save_pdf (bool): Whether to save the PDF plot (default: True).

        Raises:
            Any exceptions raised by internal methods called (`save_fits`, `plot_image`) are propagated.

        Returns:
            None
        """

        if save_fits:
            self.save_fits(image=image, filename=f"{model_name}.fits", bunit=bunit, object_name=model_name)
        self.plot_image(image=image, model_name=model_name, cmap=cmap, bunit=bunit, save_pdf=save_pdf)
    ###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###