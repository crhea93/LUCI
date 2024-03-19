.. _wavelength_corrections:

Wavelength Correction
=====================

In order to ensure that our velocity measurements (and thus broadening measurements)
are as accurate as possible, we need to add two velocity corrections:
1 - Heliocentric correction
2 - Skyline correction

As of now, only the heliocentric corrections has been applied!

Heliocentric Correction
-----------------------
This is the correction due to our movement within our own galaxy. We use the `astropy`
function `radial_velocity_correction` by passing it the location of the CFHT,
the RA/DEC of the target, and the time of the observation. This function simply outputs
the velocity in units of km/s.


Skyline Correction
------------------
Unfortunately, the wavelength must be calibrated due to slight issues with the SITELLE
instrument itself. Thankfully, this is relatively easy to do. In order to calculate
the wavelength calibration in each pixel (or spaxel if you prefer), we fit the skylines
in the pixel. OK... that would actually take way to long, so instead we split the
cube into 100x100 bins and fit the sky lines in each bin.

Please see the example notebook entitled `skyline_correction.ipynb`.
