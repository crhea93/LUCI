.. _howluciworks:

How LUCI Works
==============

LUCI is a general purpose line fitting pipeline designed to unveil the inner workings
of how fits of SITELLE IFU datacubes are calculated. In this section, I will
explain what we are trying to calculate, how it is calculated, and how you can
personalize the fits.

What we calculate
-----------------
The three primary quantities of interest are the amplitude of the line, the position
of the line (often described as the velocity and quoted in km/s), and the broadening
of the line (often described as the velocity dispersion).

Velocity
^^^^^^^^
The velocity of a line is calculated using the following equation:


Velocity Dispersion
^^^^^^^^^^^^^^^^^^^
The velocity dispersion of a line is calculated using the following equation:


How we calculate
----------------

Initial Guess
^^^^^^^^^^^^^
Having a good initial guess is crucial for the success (and speed) of the fitting algorithm.
In order to obtain a good initial guess for the fit parameters (line amplitude, line position,
and line broadening), we apply a machine learning technique described in `Rhea et al. 2020a <https://arxiv.org/abs/2008.08093>`_
(disclosure: the author of this code is also the author of this paper). The method
uses pre-trained convolutional neural networks to estimate the velocity and broadening
of the line in km/s. These are then translated into the line position and broadening. Next,
the amplitude is taken as the height of the line corresponding to the shifted line position.
We note that the machine learning model has only been trained to obtain velocities
between -500 and 500 km/s. Similarly, the model was trained to obtain broadening
values between 10 and 200 km/s.

Since we understand that machine learning is not everyone's cup of tea, we have
an alternative method to calculate the initial guesses.

Fitting Function
^^^^^^^^^^^^^^^^

Available Models
^^^^^^^^^^^^^^^^

Uncertainty Estimates
^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 2
