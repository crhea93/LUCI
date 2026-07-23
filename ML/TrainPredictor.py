"""
Create the reference spectrum and train the initial-guess neural network for a SITELLE filter.

This is the LUCI-native version of the recipe in https://github.com/sitelle-signals/Pamplemousse
(Rhea et al. 2020a, https://arxiv.org/abs/2008.08093). The original notebooks needed ORBS to
synthesise the training spectra and a connection to the 3MdB database to sample line ratios;
everything here is done with `LUCI.LuciSim.Spectrum` instead, so the only requirements are the
normal LUCI environment plus tensorflow/keras.

Running this produces the two files LUCI looks for at fit time:

    ML/Reference-Spectrum-R<resolution>-<filter>.fits   <- defines the interpolation axis
    ML/R<resolution>-PREDICTOR-I-<filter>/              <- the CNN (keras SavedModel)

and, if --mdn is given, the mixture density network used when `mdn=True`:

    ML/R<resolution>-PREDICTOR-I-MDN-<filter>/R<resolution>-PREDICTOR-I-MDN-<filter>.*

Example
-------
    python ML/TrainPredictor.py --filter SN4 --resolution 5000 --num-spectra 50000 --mdn

Before running this for a *new* filter, make sure the filter has been added to the dispatch
tables listed in `docs/source/newFilter.rst` -- in particular `LUCI.LuciSim.Spectrum` (step size
and folding order) and `LUCI.LuciUtility.read_in_reference_spectrum` (the clip applied to the
reference axis, which sets the input length of the network).
"""
import argparse
import datetime
import os
import sys

import numpy as np
from astropy.io import fits
from scipy import interpolate
from tqdm import tqdm

LUCI_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, LUCI_PATH)

from LUCI.LuciSim import Spectrum  # noqa: E402
from LUCI.LuciUtility import read_in_reference_spectrum  # noqa: E402

# Emission lines that fall inside each filter. This mirrors the `filter_line` dictionary in
# `LuciBase.Luci.create_slices`.
FILTER_LINES = {
    'SN1': ['OII3726', 'OII3729'],
    'SN2': ['OIII5007', 'OIII4959', 'Hbeta'],
    'SN3': ['Halpha', 'NII6583', 'NII6548', 'SII6716', 'SII6731'],
    'SN4': ['Halpha', 'NII6583', 'NII6548'],
}


def sample_amplitudes(filter_, rng):
    """
    Draw a set of line amplitudes for one synthetic spectrum.

    Rhea et al. 2020a sampled these from the 3MdB photoionisation/shock model grids. We instead
    sample the astrophysically relevant ratios directly over the range those grids span, which
    keeps the script self contained. The absolute scale is irrelevant -- every spectrum is
    normalised by its maximum before it reaches the network.

    Args:
        filter_: SITELLE filter (e.x. 'SN4')
        rng: numpy random generator

    Return:
        List of amplitudes ordered as FILTER_LINES[filter_]
    """
    if filter_ in ('SN3', 'SN4'):
        # NII6583/Halpha spans HII regions (~0.05) through LINERs, SNRs and AGN (~2)
        nii = 10 ** rng.uniform(-1.3, 0.3)
        ampls = [1.0, nii, nii / 3]  # Halpha, NII6583, NII6548 -- the doublet ratio is fixed
        if filter_ == 'SN3':
            sii = 10 ** rng.uniform(-1.3, 0.3)  # SII6716/Halpha
            ampls += [sii, sii * rng.uniform(0.45, 1.45)]  # the SII ratio tracks the density
        return ampls
    elif filter_ == 'SN2':
        oiii = 10 ** rng.uniform(-1.0, 1.1)  # OIII5007/Hbeta
        return [oiii, oiii / 3, 1.0]  # OIII5007, OIII4959, Hbeta
    elif filter_ == 'SN1':
        return [1.0, rng.uniform(0.35, 1.5)]  # OII3726, OII3729 -- the ratio tracks the density
    else:
        raise ValueError('No amplitude sampler defined for filter %s. Add one to FILTER_LINES '
                         'and sample_amplitudes in ML/TrainPredictor.py.' % filter_)


def make_spectrum(lines, ampls, velocity, broadening, filter_, resolution, snr, n_steps=None):
    """
    Build one synthetic spectrum, optionally forcing the number of steps.

    `LuciSim.Spectrum` derives the number of steps from the resolution as
    1.20671 * R / (order + 0.5). That comes out ~20% below both CFHT's published relation for
    SITELLE (N_steps = 1.20671^2 * R / (order + 0.5), which reproduces their quoted 0.094*R for
    SN4, 0.108*R for SN5 and 0.173*R for SN6) and the channel counts in the reference spectra
    shipped with LUCI. Passing `n_steps` explicitly -- with the STEPNB value read straight out of
    the header of the cube you intend to fit -- sidesteps the question entirely and guarantees the
    reference axis is sampled the same way your data are.

    Args:
        lines: List of lines to model
        ampls: List of amplitudes
        velocity: Velocity in km/s
        broadening: Broadening in km/s
        filter_: SITELLE filter (e.x. 'SN4')
        resolution: Spectral resolution
        snr: Signal to noise ratio
        n_steps: Number of steps; if None it is derived from the resolution

    Return:
        axis in cm-1 and the spectrum
    """
    spec = Spectrum(lines, 'sincgauss', ampls, velocity, broadening, filter_, resolution, snr)
    if n_steps is not None:
        spec.n_steps = int(n_steps)
        spec.calc_sinc_width()  # The sinc width depends on the number of steps
    return spec.create_spectrum()


def create_reference_spectrum(filter_, resolution, output_path, n_steps=None):
    """
    Create the reference spectrum for a filter/resolution pair.

    Only the *axis* of this spectrum matters -- it is the grid that every real and synthetic
    spectrum gets interpolated onto, and therefore it fixes the input length of the network.
    The fluxes are only there so the file can be inspected by eye. We use a single noisy
    Halpha-complex-like spectrum with zero velocity, which is what the shipped SN1/SN2/SN3
    reference spectra contain.

    Args:
        filter_: SITELLE filter (e.x. 'SN4')
        resolution: Spectral resolution of the cube to be fit
        output_path: Where to write the FITS file
        n_steps: Number of steps; if None it is derived from the resolution

    Return:
        Full (unclipped) axis in cm-1
    """
    lines = FILTER_LINES[filter_]
    rng = np.random.default_rng(42)
    axis, spectrum = make_spectrum(lines, sample_amplitudes(filter_, rng), 0, 10,
                                   filter_, resolution, 30, n_steps)

    col1 = fits.Column(name='Wavenumber', format='E', array=axis)
    col2 = fits.Column(name='Flux', format='E', array=spectrum)
    hdu = fits.BinTableHDU.from_columns(fits.ColDefs([col1, col2]))

    hdr = fits.Header()
    hdr['TIME'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hdr['VELOCITY'] = 0
    hdr['BROADEN'] = 10
    hdr['THETA'] = 11.96
    hdr['RES'] = resolution
    hdr['FILTER'] = filter_
    hdr['SNR'] = 30
    hdr['COMMENT'] = "Reference Spectrum"
    fits.HDUList([fits.PrimaryHDU(header=hdr), hdu]).writeto(output_path, overwrite=True)
    print('  Wrote %s (%i channels)' % (output_path, len(axis)))
    return axis


def create_training_set(filter_, resolution, wavenumbers_syn, num_spectra, n_steps=None, seed=0):
    """
    Create the synthetic training set.

    Each spectrum is built with a random velocity, broadening, set of line ratios, SNR and
    (slightly perturbed) resolution, then interpolated onto `wavenumbers_syn` and normalised by
    its maximum -- exactly the preprocessing `LuciFit.Fit.interpolate_spectrum` applies to real
    spectra at fit time.

    Args:
        filter_: SITELLE filter (e.x. 'SN4')
        resolution: Nominal spectral resolution
        wavenumbers_syn: Clipped reference axis in cm-1
        num_spectra: Number of synthetic spectra to create
        n_steps: Number of steps; if None it is derived from the resolution
        seed: Random seed

    Return:
        counts (num_spectra, len(wavenumbers_syn)) and labels (num_spectra, 2) [velocity, broadening]
    """
    rng = np.random.default_rng(seed)
    lines = FILTER_LINES[filter_]
    counts = np.zeros((num_spectra, len(wavenumbers_syn)), dtype=np.float32)
    labels = np.zeros((num_spectra, 2), dtype=np.float32)

    for i in tqdm(range(num_spectra)):
        # The network is only ever asked for velocities in [-500, 500] km/s and broadenings in
        # [10, 200] km/s (see docs/source/howLuciWorks.rst), so we train over exactly that range
        velocity = rng.uniform(-500, 500)
        broadening = rng.uniform(10, 200)
        snr = rng.uniform(5, 100)
        # Let the resolution wander a little -- real cubes never sit exactly at the nominal value
        res = rng.uniform(0.95 * resolution, resolution)
        axis, spectrum = make_spectrum(lines, sample_amplitudes(filter_, rng), velocity,
                                       broadening, filter_, res, snr, n_steps)
        f = interpolate.interp1d(axis, spectrum, kind='slinear', fill_value='extrapolate')
        interpolated = f(wavenumbers_syn)
        counts[i] = interpolated / np.max(interpolated)
        labels[i] = (velocity, broadening)

    return counts, labels


def build_cnn(input_length):
    """
    The convolutional network of Rhea et al. 2020a. Two convolutional layers, a max pooling
    layer, and two hidden layers ending in a linear velocity/broadening output.

    Args:
        input_length: Length of the input spectrum

    Return:
        Compiled keras model
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, InputLayer, Flatten, Dropout, Conv1D, MaxPooling1D
    from tensorflow.keras.optimizers.legacy import Adam

    model = Sequential([
        InputLayer(batch_input_shape=(None, input_length, 1)),
        Conv1D(kernel_initializer='normal', activation='relu', padding='same', filters=4, kernel_size=4),
        Conv1D(kernel_initializer='normal', activation='relu', padding='same', filters=16, kernel_size=2),
        MaxPooling1D(pool_size=4),
        Flatten(),
        Dropout(0.2),
        Dense(units=256, kernel_initializer='normal', activation='relu'),
        Dense(units=512, kernel_initializer='normal', activation='relu'),
        Dense(2, activation='linear'),
    ])
    model.compile(optimizer=Adam(learning_rate=0.0007, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                  loss='mean_squared_error', metrics=['mse', 'mae'])
    return model


def train(model, counts, labels, max_epochs, batch_size, loss=None):
    """
    Split the synthetic data 70/20/10 into training/validation/test, fit, and report the
    residual scatter on the test set.

    Args:
        model: Compiled keras model
        counts: Synthetic spectra
        labels: [velocity, broadening] labels
        max_epochs: Maximum number of epochs
        batch_size: Batch size
        loss: Unused -- kept so the caller can stay symmetric between the CNN and MDN

    Return:
        The trained model
    """
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    counts = counts.reshape(counts.shape[0], counts.shape[1], 1)
    train_div = int(0.7 * len(counts))
    valid_div = int(0.9 * len(counts))

    callbacks = [
        ReduceLROnPlateau(monitor='loss', factor=0.5, min_delta=0.009, patience=2,
                          min_lr=0.00008, mode='min', verbose=2),
        EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=4, verbose=2, mode='min'),
    ]
    model.fit(counts[:train_div], labels[:train_div],
              validation_data=(counts[train_div:valid_div], labels[train_div:valid_div]),
              epochs=max_epochs, batch_size=batch_size, verbose=2, callbacks=callbacks)

    predictions = model(counts[valid_div:], training=False)
    if hasattr(predictions, 'mean'):  # The MDN returns a distribution rather than point estimates
        predictions = predictions.mean()
    residuals = np.array(predictions) - labels[valid_div:]
    print('  Test set velocity residual   : %.2f +/- %.2f km/s' % (np.mean(residuals[:, 0]), np.std(residuals[:, 0])))
    print('  Test set broadening residual : %.2f +/- %.2f km/s' % (np.mean(residuals[:, 1]), np.std(residuals[:, 1])))
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--filter', required=True, help="SITELLE filter (e.x. 'SN4')")
    parser.add_argument('--resolution', required=True, type=int,
                        help='Spectral resolution of the cube you want to fit (e.x. 5000)')
    parser.add_argument('--n-steps', type=int, default=None,
                        help='Number of steps of the synthetic spectra. Strongly recommended: use '
                             'the STEPNB value from the header of the cube you want to fit. '
                             'Defaults to LuciSim deriving it from the resolution.')
    parser.add_argument('--num-spectra', type=int, default=50000,
                        help='Number of synthetic spectra to generate (default 50000)')
    parser.add_argument('--epochs', type=int, default=10, help='Maximum number of epochs (default 10)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default 32)')
    parser.add_argument('--mdn', action='store_true',
                        help='Also train the mixture density network used when mdn=True')
    parser.add_argument('--output-dir', default=os.path.join(LUCI_PATH, 'ML'),
                        help='Where to write the reference spectrum and networks (default ML/)')
    args = parser.parse_args()

    if args.filter not in FILTER_LINES:
        parser.error('Unknown filter %s. Add it to FILTER_LINES and sample_amplitudes first.' % args.filter)

    tag = 'R%i-%s' % (args.resolution, args.filter)
    ref_path = os.path.join(args.output_dir, 'Reference-Spectrum-%s.fits' % tag)

    print('# -- Creating the reference spectrum -- #')
    create_reference_spectrum(args.filter, args.resolution, ref_path, args.n_steps)
    # Clip the reference axis exactly the way LUCI does when it reads a cube, so that the network
    # input length matches what `LuciFit` will feed it
    wavenumbers_syn, _ = read_in_reference_spectrum(ref_path, {'FILTER': args.filter})
    print('  Network input length: %i channels (%.1f - %.1f cm-1)'
          % (len(wavenumbers_syn), wavenumbers_syn[0], wavenumbers_syn[-1]))

    print('# -- Creating %i synthetic spectra -- #' % args.num_spectra)
    counts, labels = create_training_set(args.filter, args.resolution, wavenumbers_syn,
                                         args.num_spectra, args.n_steps)

    print('# -- Training the CNN -- #')
    model = train(build_cnn(len(wavenumbers_syn)), counts, labels, args.epochs, args.batch_size)
    cnn_path = os.path.join(args.output_dir, 'R%i-PREDICTOR-I-%s' % (args.resolution, args.filter))
    # 'tf' (SavedModel) is what `keras.models.load_model` reads back in `LuciFit.Fit.get_ML_model`
    model.save(cnn_path, save_format='tf')
    print('  Saved %s' % cnn_path)

    if args.mdn:
        from LUCI.LuciNetwork import create_MDN_model, negative_loglikelihood
        print('# -- Training the MDN -- #')
        mdn_model = train(create_MDN_model(len(wavenumbers_syn), negative_loglikelihood),
                          counts, labels, args.epochs, args.batch_size)
        # LUCI rebuilds the MDN architecture and calls `load_weights`, so we save weights (not a
        # SavedModel) using the doubled-up path convention it expects
        mdn_dir = os.path.join(args.output_dir, 'R%i-PREDICTOR-I-MDN-%s' % (args.resolution, args.filter))
        os.makedirs(mdn_dir, exist_ok=True)
        mdn_path = os.path.join(mdn_dir, 'R%i-PREDICTOR-I-MDN-%s' % (args.resolution, args.filter))
        mdn_model.save_weights(mdn_path, save_format='tf')
        print('  Saved %s' % mdn_path)


if __name__ == '__main__':
    main()
