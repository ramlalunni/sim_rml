from astropy.io import fits
import yaml
from types import SimpleNamespace
from pathlib import Path
import warnings

###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###
class FITSHeaderWarning(UserWarning):
    pass
###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###

###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###
def process_inputs(yaml_file:str = "inputs.yaml") -> SimpleNamespace:
    """
    Reads the input YAML file and returns a SimpleNamespace with each section as attributes.

    Args:
        yaml_file (str): Path to the YAML input file.

    Raises:
        FileNotFoundError: If the specified YAML file does not exist.

    Returns:
        SimpleNamespace: An object with keys corresponding to top-level YAML sections.
                         Each key is itself a namespace (e.g., config.preset_model.fov_arcsec).
    """

    if not yaml_file.endswith(".yaml"):
        yaml_file += ".yaml"

    yaml_path = Path(yaml_file)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"YAML input file not found: {yaml_file}")

    with open(yaml_path, "r") as yf:
        data = yaml.safe_load(yf)

    def to_namespace(d: dict) -> SimpleNamespace:
        return SimpleNamespace(**{
            k: to_namespace(v) if isinstance(v, dict) else v
            for k, v in d.items()
        })

    return to_namespace(data)
###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###

###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###
def check_user_FITS_file(fits_file):
    """
    Check if the provided FITS file has a valid header and data structure for simalma.

    Args:
        fits_file (str): Path to the FITS file to be checked.

    Raises:
        ValueError: If the FITS file does not have the required header or data structure.
        FITSHeaderWarning: If optional header elements are missing.

    Returns:
        bool: True if the FITS file is valid for simalma, False otherwise.
    """

    try:
        with fits.open(fits_file) as hdul:
            if len(hdul) == 0 or not hdul[0].header:
                raise ValueError("FITS file does not have a primary header.")

            required_mandatory_keywords = ["SIMPLE", "BITPIX", "NAXIS"]
            for keyword in required_mandatory_keywords:
                if keyword not in hdul[0].header:
                    raise ValueError(f"Missing required header element: {keyword}")

            required_optional_keywords = ["CRPIX1", "CDELT1", "CUNIT1", "CTYPE1", "CRVAL1", "CRPIX2", "CDELT2", "CUNIT2", "CTYPE2", "CRVAL2", "CRPIX3", "CDELT3", "CUNIT3", "CTYPE3", "CRVAL3"]
            for keyword in required_optional_keywords:
                if keyword not in hdul[0].header:
                    warnings.warn(f"Missing header element: {keyword} in the input FITS file. Please update your FITS file or provide it in the run_simalma() call as needed. See `simulate_ALMA_observations.run_simalma` for details. Missing this information might lead to errors/wrong results.", FITSHeaderWarning)

            if hdul[0].data is None:
                raise ValueError("No data found in the primary HDU.")

    except Exception as e:
        raise ValueError(f"Invalid FITS file: {e}")

    return True
###~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~###

