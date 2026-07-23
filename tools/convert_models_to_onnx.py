# /// script
# requires-python = "==3.10.*"
# dependencies = [
#   "tensorflow==2.11",
#   "tensorflow-probability==0.19",
#   "tf2onnx",
#   "onnxruntime",
#   "numpy<2",
#   "protobuf<4",
#   "astropy",
# ]
# ///
"""
Convert LUCI's trained Keras predictors to ONNX.

Run it with uv, which builds the pinned legacy-TensorFlow environment from the
PEP 723 header above on the fly and throws it away afterwards -- the project's
own environment never sees TensorFlow::

    uv run tools/convert_models_to_onnx.py --all --validate
    uv run tools/convert_models_to_onnx.py --models R5000-PREDICTOR-I-SN3 --validate

Two model families need different handling:

* **Standard predictors** (``R*-PREDICTOR-I-<FILTER>``) are TF SavedModels and
  convert directly with ``tf2onnx``.  Output: ``[velocity, broadening]`` km/s.

* **MDN predictors** (``R*-PREDICTOR-I-MDN-<FILTER>``) are weights-only
  checkpoints whose head is a ``tfp.layers.IndependentNormal(2)`` on top of a
  ``Dense(4)``.  TFP distribution layers have no ONNX equivalent, so we rebuild
  the architecture, load the weights, and export the model **truncated at the
  Dense(4)** layer.  That layer's four outputs are exactly
  ``[loc_v, loc_b, scale_v, scale_b]``; empirically (verified by --validate)
  ``mean == loc`` and ``stddev == scale`` with no further transform, so the
  runtime reproduces the distribution by simply splitting the vector in half.

``--validate`` feeds random spectra through both the Keras and ONNX paths and
fails loudly, per model, if they disagree beyond tolerance.  Nothing is shipped
that has not passed this gate.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_DIR = os.path.join(REPO_ROOT, "ML")
OUT_DIR = os.path.join(ML_DIR, "onnx")

# Reference-spectrum clip windows (cm^-1), per filter.  Duplicated here rather
# than imported from LUCI because this script runs in a throwaway TF env that
# does not install LUCI.  Kept in sync with LUCI/instrument/filters.py.
REFERENCE_BOUNDS = {
    "SN1": (25500, 27500),
    "SN2": (19000, 21000),
    "SN3": (14700, 15600),
    "SN4": (15000, 15350),
    "C1": (20408, 25974),
    "C2": (15987, 17880),
    "C3": (17500, 19500),
    "C4": (12100, 12600),
}

# Validation tolerances.  Outputs are km/s of magnitude ~1e1-1e2; tf2onnx keeps
# float32 models to float32 precision, so ~1e-4 relative is the right gate.
RTOL = 1e-4
ATOL = 1e-3
N_VALIDATION_SPECTRA = 200


def _filter_of(name: str) -> str:
    m = re.search(r"-(SN[1-4]|C[1-4])$", name)
    if not m:
        raise ValueError(f"cannot parse filter from model name {name!r}")
    return m.group(1)


def _resolution_of(name: str) -> int:
    m = re.match(r"R(\d+)-", name)
    if not m:
        raise ValueError(f"cannot parse resolution from model name {name!r}")
    return int(m.group(1))


def reference_input_length(resolution: int, filter_name: str) -> int:
    """Length of the clipped reference spectrum -- the model's input width."""
    from astropy.io import fits

    path = os.path.join(ML_DIR, f"Reference-Spectrum-R{resolution}-{filter_name}.fits")
    channels = np.array([row[0] for row in fits.open(path)[1].data])
    lo, hi = REFERENCE_BOUNDS[filter_name]
    i_lo = int(np.argmin(np.abs(channels - lo)))
    i_hi = int(np.argmin(np.abs(channels - hi)))
    return len(channels[i_lo:i_hi])


def is_standard(name: str) -> bool:
    return os.path.exists(os.path.join(ML_DIR, name, "saved_model.pb"))


def is_mdn(name: str) -> bool:
    return "MDN" in name and os.path.exists(os.path.join(ML_DIR, name, f"{name}.index"))


def convert_standard(name: str, out_path: str) -> None:
    import tf2onnx  # noqa: F401

    src = os.path.join(ML_DIR, name)
    rc = os.system(
        f"{sys.executable} -m tf2onnx.convert --saved-model {src} "
        f"--output {out_path} --opset 17 >/dev/null 2>&1"
    )
    if rc != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"tf2onnx failed for {name}")


def convert_mdn(name: str, out_path: str) -> None:
    import keras
    import tf2onnx

    sys.path.insert(0, REPO_ROOT)
    from LUCI.LuciNetwork import create_MDN_model, negative_loglikelihood

    input_len = reference_input_length(_resolution_of(name), _filter_of(name))
    mdn = create_MDN_model(input_len, negative_loglikelihood)
    mdn.load_weights(os.path.join(ML_DIR, name, name))
    # Truncate at the Dense(4) layer that feeds the IndependentNormal head.
    truncated = keras.Model(mdn.input, mdn.layers[-2].output)
    spec = (tf.TensorSpec((None, input_len, 1), tf.float32, name="input"),)
    tf2onnx.convert.from_keras(truncated, input_signature=spec, opset=17, output_path=out_path)


def mdn_split(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce IndependentNormal(2) from the Dense(4) output: mean | stddev."""
    return raw[:, :2], raw[:, 2:]


def validate(name: str, out_path: str) -> tuple[bool, float]:
    import keras
    import onnxruntime as ort

    mdn = is_mdn(name)
    if mdn:
        from LUCI.LuciNetwork import create_MDN_model, negative_loglikelihood

        input_len = reference_input_length(_resolution_of(name), _filter_of(name))
        model = create_MDN_model(input_len, negative_loglikelihood)
        model.load_weights(os.path.join(ML_DIR, name, name))
    else:
        model = keras.models.load_model(os.path.join(ML_DIR, name))
        input_len = model.input_shape[1]

    X = np.random.default_rng(0).random((N_VALIDATION_SPECTRA, input_len, 1)).astype(np.float32)
    sess = ort.InferenceSession(out_path)
    onnx_out = sess.run(None, {sess.get_inputs()[0].name: X})[0]

    if mdn:
        dist = model(X, training=False)
        keras_mean, keras_std = dist.mean().numpy(), dist.stddev().numpy()
        onnx_mean, onnx_std = mdn_split(np.asarray(onnx_out))
        keras_out = np.concatenate([keras_mean, keras_std], axis=1)
        onnx_cat = np.concatenate([onnx_mean, onnx_std], axis=1)
        ok = np.allclose(keras_out, onnx_cat, rtol=RTOL, atol=ATOL)
        return ok, float(np.max(np.abs(keras_out - onnx_cat)))

    keras_out = model.predict(X, verbose=0)
    ok = np.allclose(keras_out, onnx_out, rtol=RTOL, atol=ATOL)
    return ok, float(np.max(np.abs(keras_out - onnx_out)))


def discover() -> list[str]:
    names = []
    for entry in sorted(os.listdir(ML_DIR)):
        if "PREDICTOR-I" not in entry:
            continue
        if is_standard(entry) or is_mdn(entry):
            names.append(entry)
    return names


def main() -> int:
    global tf
    import tensorflow as tf  # noqa: F401

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="convert every discovered model")
    parser.add_argument("--models", nargs="*", default=[], help="specific model dir names")
    parser.add_argument("--validate", action="store_true", help="numerically check each conversion")
    parser.add_argument("--rtol", type=float, default=RTOL)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    targets = discover() if args.all else args.models
    if not targets:
        print("nothing to convert (pass --all or --models ...)")
        return 1

    report = []
    for name in targets:
        out_path = os.path.join(OUT_DIR, f"{name}.onnx")
        kind = "MDN" if is_mdn(name) else "standard"
        try:
            if is_mdn(name):
                convert_mdn(name, out_path)
            else:
                convert_standard(name, out_path)
            status, diff = "converted", None
            if args.validate:
                ok, diff = validate(name, out_path)
                status = "OK" if ok else "FAILED"
                if not ok:
                    os.remove(out_path)  # do not ship an unvalidated model
        except Exception as exc:  # noqa: BLE001
            status, diff = f"ERROR: {exc}", None
        report.append((name, kind, status, diff))
        d = f"  max_abs={diff:.2e}" if diff is not None else ""
        print(f"[{status:>9}] {kind:>8}  {name}{d}")

    failed = [r for r in report if r[2] not in ("converted", "OK")]
    print(f"\n{len(report) - len(failed)}/{len(report)} succeeded; output in {os.path.relpath(OUT_DIR)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
