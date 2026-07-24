"""PCA model of the sky background across the field."""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as spi
from astropy.io import fits
from sklearn import decomposition
from sklearn.model_selection import train_test_split

from LUCI.background.detection import find_background_pixels
from LUCI.instrument.filters import pca_scale_indices


def create_background_subspace(
    cube,
    x_min=100,
    x_max=1900,
    y_min=100,
    y_max=1900,
    bkg_image="deep",
    n_components=50,
    n_components_keep=None,
    sigma_threshold=0.1,
    npixels=10,
    bkg_algo="detect_source",
    interpolation="nn",
):
    """
    This function will create a subspace of principal components describing the background emission. It will then interpolate
    the background eigenvectors over the entire field. Please see our paper () describing this methodology in detail.
    After running this code, `LUCI` will create a fits file containing the PCA coefficients for each pixel in the FOV which
    can then be used in other fitting functions by setting the optional argument `bkg` to `PCA`. `LUCI` will also save a
    numpy file with the principal components.
    We also create 3 plots that are saved in `cube.output_dir`:
     - Map of background pixels in FOV
     - Visualization of first 10 principal components and mean
     - Scree plot

    Args:
        x_min: Minimum x value (image coordinates) for segmentation region (Default 100)
        x_max: Maximum x value (image coordinates) for segmentation region (Default 1900)
        y_min: Minimum y value (image coordinates) for segmentation region (Default 100)
        y_max: Maximum y value (image coordinates) for segmentation region (Default 1900)
        bkg_image: 2D image used for background thresholding (Default deep image). Pass fits file.
        n_components: Number of principal components to calculate (Default 50)
        n_components_keep: Number of principal components to keep (Default n_components)
        sigma_threshold: Threshold parameter for determining the background (default 0.1)
        npixels: Minimum number of connected pixels in a detected group (default 10)
        bkg_algo: Background algorithm to use (default 'sourece_detect'; options: 'source_detect', 'threshold')
        interpolation: Scheme for interpolation (default 'nn'; options: 'nn', 'linear', 'nearest')

    Return:
        PCA_coeffs: Fits file containing the PCA coefficients over the FOV
        PCA_eigenspectra: Numpy file containing the PCA eigenspectra

    """
    if n_components_keep is None:
        n_components_keep = n_components
    if bkg_image == "deep":
        cube.create_deep_image()
        background_image = cube.deep_image.T[x_min:x_max, y_min:y_max]
    else:
        background_image = fits.open(bkg_image)[0].data[x_min:x_max, y_min:y_max]
    # Find background pixels and save map in cube.output_dir
    idx_bkg, idx_src = find_background_pixels(
        background_image,
        cube.output_dir,
        sigma_threshold=sigma_threshold,
        npixels=npixels,
        bkg_algo=bkg_algo,
        filter_=cube.filter,
    )  # Get IDs of background and source pixels
    max_spectral = None  # Initialize
    min_spectral = None  # Initialize
    if cube.filter == "SN3":
        max_spectral = len(
            cube.spectrum_axis
        )  # np.argmin(np.abs([1e7 / wavelength - 646 for wavelength in cube.spectrum_axis]))
        min_spectral = 0  # np.argmin(np.abs([1e7 / wavelength - 678 for wavelength in cube.spectrum_axis]))
    elif cube.filter == "SN2":
        max_spectral = len(
            cube.spectrum_axis
        )  # np.argmin(np.abs([1e7 / wavelength - 480 for wavelength in cube.spectrum_axis]))
        min_spectral = 0  # np.argmin(np.abs([1e7 / wavelength - 505 for wavelength in cube.spectrum_axis]))
    elif cube.filter == "SN1":
        max_spectral = len(
            cube.spectrum_axis
        )  # np.argmin(np.abs([1e7 / wavelength - 360 for wavelength in cube.spectrum_axis]))
        min_spectral = 0  # np.argmin(np.abs([1e7 / wavelength - 380 for wavelength in cube.spectrum_axis]))
    elif cube.filter == "SN4":
        max_spectral = len(cube.spectrum_axis)
        min_spectral = 0
    else:
        print(
            "We have yet to implement this algorithm for this filter. So far we have implemented it for SN4, SN3, SN2, and SN1"
        )
        print("Terminating Program")
        quit()
    # Check if there are not enough components
    if len(cube.cube_final[100, 100, min_spectral:max_spectral]) < n_components:
        n_components = len(cube.cube_final[100, 100, min_spectral:max_spectral])
        if n_components_keep > n_components:
            n_components_keep = n_components
    bkg_spectra = [
        cube.cube_final[x_min + index[0], y_min + index[1], min_spectral:max_spectral] for index in idx_bkg
    ]  # Get background pixels
    # Calculate scaling factors and normalize
    if cube.filter == "SN3":
        min_spectral_scale = np.argmin(
            np.abs([1e7 / wavelength - 675 for wavelength in cube.spectrum_axis[min_spectral:max_spectral]])
        )
        max_spectral_scale = np.argmin(
            np.abs([1e7 / wavelength - 670 for wavelength in cube.spectrum_axis[min_spectral:max_spectral]])
        )
    elif cube.filter == "SN2":
        min_spectral_scale = np.argmin(
            np.abs([1e7 / wavelength - 505 for wavelength in cube.spectrum_axis[min_spectral:max_spectral]])
        )
        max_spectral_scale = np.argmin(
            np.abs([1e7 / wavelength - 480 for wavelength in cube.spectrum_axis[min_spectral:max_spectral]])
        )
    elif cube.filter == "SN1":
        min_spectral_scale = np.argmin(
            np.abs([1e7 / wavelength - 365 for wavelength in cube.spectrum_axis[min_spectral:max_spectral]])
        )
        max_spectral_scale = np.argmin(
            np.abs([1e7 / wavelength - 360 for wavelength in cube.spectrum_axis[min_spectral:max_spectral]])
        )
    elif cube.filter == "SN4":
        # The SN4 pass band is only 652-665 nm, so we scale on the line free red end of it
        min_spectral_scale = np.argmin(
            np.abs([1e7 / wavelength - 664.5 for wavelength in cube.spectrum_axis[min_spectral:max_spectral]])
        )
        max_spectral_scale = np.argmin(
            np.abs([1e7 / wavelength - 661 for wavelength in cube.spectrum_axis[min_spectral:max_spectral]])
        )
    else:
        print(
            "We have yet to implement this algorithm for this filter. So far we have implemented it for SN4, SN3, SN2, and SN1"
        )
        print("Terminating Program")
        quit()
    bkg_spectra = [
        bkg_spectrum / np.nanmax(bkg_spectrum[min_spectral_scale:max_spectral_scale]) for bkg_spectrum in bkg_spectra
    ]
    bkg_spectra = [bkg_spectrum / np.max(bkg_spectrum) for bkg_spectrum in bkg_spectra]
    # Remove outliers
    # outlier_predictions = IsolationForest(random_state=0).fit_predict(bkg_spectra)  # Outliers have a value of -1 and inliers have a value of 1
    # outlier_predictions = np.where(outlier_predictions == 1)
    # outlier_predictions = ma.masked_where(outlier_predictions > 0, outlier_predictions).mask
    # bkg_spectra = np.array(bkg_spectra)[outlier_predictions[:]]
    # Calculate n most important components
    spectral_axis_nm = 1e7 / cube.spectrum_axis[min_spectral:max_spectral]
    pca = decomposition.PCA(n_components=n_components)  # Call pca
    pca.fit(bkg_spectra)  # Fit using background spectra
    pickle.dump(pca, open(os.path.join(cube.output_dir, "pca_%s.pkl" % cube.filter), "wb"))
    BkgTransformedPCA = pca.transform(bkg_spectra)[:, :n_components_keep]  # Apply on background spectra
    # Plot the normalized primary components
    plt.figure(figsize=(18, 16))
    l = plt.plot(spectral_axis_nm, pca.mean_ / np.max(pca.mean_) - 2, linewidth=3)  # plot the mean first
    c = l[0].get_color()
    plt.text(670, -0.9, "mean emission", color=c, fontsize="xx-large")
    shift = 2
    for i in range(10):  # Plot first 10 components
        l = plt.plot(spectral_axis_nm, pca.components_[i] / np.max(pca.components_[i]) + (i * shift), linewidth=3)
        c = l[0].get_color()
        plt.text(670, i * shift + 0.3, "component %i" % (i + 1), color=c, fontsize="xx-large")
    plt.xlabel("nm", fontsize=24)
    plt.ylabel("Normalized Emission + Offset", fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.savefig(os.path.join(cube.output_dir, "PCA_components_normalized_%s.png" % cube.filter))
    # Plot the primary components
    plt.figure(figsize=(18, 16))
    l = plt.plot(spectral_axis_nm, pca.mean_ - 2, linewidth=3)  # plot the mean first
    c = l[0].get_color()
    plt.text(670, -0.9, "mean emission", color=c, fontsize="xx-large")
    shift = 2
    for i in range(10):  # Plot first 10 components
        l = plt.plot(spectral_axis_nm, pca.components_[i] + (i * shift), linewidth=3)
        c = l[0].get_color()
        plt.text(670, i * shift + 0.3, "component %i" % (i + 1), color=c, fontsize="xx-large")
    plt.xlabel("nm", fontsize=24)
    plt.ylabel("Normalized Emission + Offset", fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.savefig(os.path.join(cube.output_dir, "PCA_components_%s.png" % cube.filter))
    # Make scree plot
    plt.figure(figsize=(14, 8))
    PC_values = np.arange(pca.n_components_)[:n_components] + 1
    explained_variance_ratio = [
        (1 / np.sum(pca.explained_variance_ratio_)) * pca.explained_variance_ratio_[i] for i in range(n_components)
    ]
    plt.plot(PC_values, explained_variance_ratio, "o-", linewidth=3)
    # plt.title('Scree Plot')
    plt.xlabel("Principal Component", fontsize=24)
    plt.ylabel("Variance Explained", fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.savefig(os.path.join(cube.output_dir, "PCA_scree_%s.png" % cube.filter))
    # Collect background and source pixels/coordinates
    # bkg_pixels = [[x_min+index[0], y_min+index[1]] for index in idx_bkg[outlier_predictions]]
    bkg_pixels = [[x_min + index[0], y_min + index[1]] for index in idx_bkg]
    src_pixels = [[x_min + index[0], y_min + index[1]] for index in idx_src]
    bkg_x = [bkg[0] for bkg in bkg_pixels]
    bkg_y = [bkg[1] for bkg in bkg_pixels]
    src_x = [src[0] for src in src_pixels]
    src_y = [src[1] for src in src_pixels]
    # Interpolate
    # interpolatedSourcePixels = None
    if interpolation in ["linear", "nearest"]:
        interpolatedSourcePixels = spi.griddata(bkg_pixels, BkgTransformedPCA, src_pixels, method=interpolation)
    else:  # Use neural network
        # Imported lazily so `import LuciBase` does not pull in TensorFlow.
        try:
            from keras.callbacks import EarlyStopping, ReduceLROnPlateau
            from keras.layers import Dense, Dropout, InputLayer
            from keras.models import Sequential
            from keras.regularizers import l2
            from tensorflow.keras.optimizers.legacy import Adam
        except ImportError as exc:
            raise ImportError(
                "interpolation='nn' trains a Keras model and requires TensorFlow, "
                "which is an optional dependency. Either install it with "
                "pip install 'luci-sitelle[ml-legacy]', or use "
                "interpolation='linear' / 'nearest', which need only scipy."
            ) from exc
        # Construct Neural Network
        X_train, X_valid, y_train, y_valid = train_test_split(
            np.column_stack((bkg_x, bkg_y)), BkgTransformedPCA[:], test_size=0.05
        )
        ### Model creation: adding layers and compilation
        hiddenActivation = "tanh"  # activation function
        input_shape = (None, 2)
        num_hidden = [200, 300]  # number of nodes in the hidden layers
        batch_size = 8  # number of data fed into model at once
        max_epochs = 100  # maximum number of interations
        lr = 1e-2  # 8e-5  # initial learning rate
        beta_1 = 0.9  # exponential decay rate  - 1st
        beta_2 = 0.999  # exponential decay rate  - 2nd
        optimizer_epsilon = 1e-08  # For the numerical stability
        early_stopping_min_delta = 0.0001
        early_stopping_patience = 12
        reduce_lr_factor = 0.75
        reuce_lr_epsilon = 0.009
        reduce_lr_patience = 4
        reduce_lr_min = 1e-7
        loss_function = "huber"  # 'mean_squared_error'
        metrics_ = ["mae", "mape"]
        model2D = Sequential(
            [
                InputLayer(batch_input_shape=input_shape),
                Dense(units=num_hidden[0], activation=hiddenActivation, kernel_regularizer=l2(0.00005)),
                Dropout(0.18),
                Dense(units=num_hidden[1], activation=hiddenActivation, kernel_regularizer=l2(0.00005)),
                Dense(n_components_keep, activation="linear"),
            ]
        )
        # Set optimizer
        optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=optimizer_epsilon, decay=0.0)
        # Set early stopping conditions
        early_stopping = EarlyStopping(
            monitor="loss", min_delta=early_stopping_min_delta, patience=early_stopping_patience, verbose=2, mode="min"
        )
        # Set learn rate reduction conditions
        reduce_lr = ReduceLROnPlateau(
            monitor="loss",
            factor=0.5,
            epsilon=reuce_lr_epsilon,
            patience=reduce_lr_patience,
            min_lr=reduce_lr_min,
            mode="min",
            verbose=2,
        )
        # Compile CNN
        model2D.compile(optimizer=optimizer, loss=loss_function, metrics=metrics_)

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
        X_valid = X_valid.reshape(X_valid.shape[0], X_valid.shape[1], 1)
        y_valid = y_valid.reshape(y_valid.shape[0], y_valid.shape[1], 1)
        history = model2D.fit(
            X_train,
            y_train,
            epochs=max_epochs,
            batch_size=batch_size,
            validation_data=(X_valid, y_valid),
            callbacks=[reduce_lr, early_stopping],
        )
        # Predict using model
        interpolatedSourcePixels = model2D.predict(np.column_stack((src_x, src_y)))
    # Construct final coefficient array, sized from the cube rather than the
    # standard detector dimensions (bug B10).
    coefficient_array = np.zeros((*cube.cube_final.shape[:2], n_components_keep))
    # coefficient_array[:] = np.nan
    for pixel_ct, pixel in enumerate(bkg_pixels):
        coefficient_array[pixel[0], pixel[1]] = BkgTransformedPCA[pixel_ct]
    for pixel_ct, pixel in enumerate(src_pixels):
        coefficient_array[pixel[0], pixel[1]] = interpolatedSourcePixels[pixel_ct]
    pickle.dump(
        coefficient_array, open(os.path.join(cube.output_dir, "pca_coefficient_array_%s.pkl" % cube.filter), "wb")
    )
    # Make coefficient maps for first 5 coefficients
    coeff_map_path = os.path.join(cube.output_dir, "PCACoefficientMaps")
    if not os.path.exists(coeff_map_path):
        os.mkdir(coeff_map_path)
    for n_component in range(n_components_keep):
        plt.figure(figsize=(18, 16))
        coeff_map = coefficient_array[:, :, n_component][
            x_min:x_max, y_min:y_max
        ]  # /np.nanmax(coefficient_array[:,:,n_component])
        plt.imshow(coeff_map.T, origin="lower", cmap="viridis")
        c_min = np.nanpercentile(coeff_map, 5)
        c_max = np.nanpercentile(coeff_map, 99.5)
        plt.title("Component %i" % (n_component + 1))
        plt.xlabel("Physical Coordinates", fontsize=24, fontweight="bold")
        plt.ylabel("Physical Coordinates", fontsize=24, fontweight="bold")
        plt.clim(c_min, c_max)
        plt.savefig(os.path.join(coeff_map_path, "component%i_%s.png" % (n_component + 1, cube.filter)))
        fits.writeto(
            os.path.join(coeff_map_path, "component%i_%s.fits" % (n_component + 1, cube.filter)),
            coefficient_array[:, :, n_component],
            cube.header,
            overwrite=True,
        )
    return BkgTransformedPCA, pca, interpolatedSourcePixels, idx_bkg, idx_src, coefficient_array
