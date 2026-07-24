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
  Dense(4)** layer.  That layer's four outputs are
  ``[loc_v, loc_b, scale_v, scale_b]``; the runtime recovers the distribution as
  ``mean = loc`` and ``stddev = softplus(scale)`` (see ``mdn_split``).

  The softplus is easy to miss: predictors whose raw scales are all large and
  positive satisfy ``softplus(x) == x`` in float32, so identity looks correct on
  them.  The SN2 MDNs emit *negative* raw scales, where identity yields a
  negative standard deviation and is off by up to 35 km/s.  ``--validate``
  caught exactly that.

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
        f"{sys.executable} -m tf2onnx.convert --saved-model {src} --output {out_path} --opset 17 >/dev/null 2>&1"
    )
    if rc != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"tf2onnx failed for {name}")


def convert_mdn(name: str, out_path: str) -> None:
    import keras
    import tf2onnx

    sys.path.insert(0, REPO_ROOT)
    from luci.ml.mdn_architecture import create_MDN_model, negative_loglikelihood

    input_len = reference_input_length(_resolution_of(name), _filter_of(name))
    mdn = create_MDN_model(input_len, negative_loglikelihood)
    mdn.load_weights(os.path.join(ML_DIR, name, name))
    # Truncate at the Dense(4) layer that feeds the IndependentNormal head.
    truncated = keras.Model(mdn.input, mdn.layers[-2].output)
    spec = (tf.TensorSpec((None, input_len, 1), tf.float32, name="input"),)
    tf2onnx.convert.from_keras(truncated, input_signature=spec, opset=17, output_path=out_path)


def mdn_split(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Reproduce ``IndependentNormal(2)`` from the Dense(4) output.

    ``mean`` is the first half verbatim; ``stddev`` is **softplus** of the second
    half -- the layer's positivity transform.  ``np.logaddexp(0, x)`` is a
    numerically stable softplus (no exp overflow for large x).

    This is worth spelling out because it is easy to get wrong: for predictors
    whose raw scale outputs are all large and positive (e.g. R5000-MDN-SN3,
    range ~[77, 430]) ``softplus(x) == x`` exactly in float32, so identity looks
    correct.  It is not -- the SN2 MDNs emit negative raw scales, where identity
    is off by up to 35 km/s.  --validate catches this.
    """
    return raw[:, :2], np.logaddexp(0.0, raw[:, 2:])


def _standard_reference(name: str):
    """
    Return ``(predict_fn, input_len)`` for a standard SavedModel.

    Prefers the low-level ``tf.saved_model.load`` serving signature over
    ``keras.models.load_model``: the latter chokes on models saved by a newer
    Keras (e.g. the SN4 predictor raises "Unrecognized keyword arguments:
    ['optional']"), while the serving signature is exactly what tf2onnx consumes,
    so it is both more robust and the more faithful reference.
    """
    loaded = tf.saved_model.load(os.path.join(ML_DIR, name))
    infer = loaded.signatures["serving_default"]
    input_spec = list(infer.structured_input_signature[1].values())[0]
    input_len = int(input_spec.shape[1])
    output_key = list(infer.structured_outputs.keys())[0]

    def predict_fn(x: np.ndarray) -> np.ndarray:
        result = infer(tf.convert_to_tensor(x))
        return result[output_key].numpy()

    return predict_fn, input_len


def validate(name: str, out_path: str) -> tuple[bool, float]:
    import onnxruntime as ort

    mdn = is_mdn(name)
    if mdn:
        from luci.ml.mdn_architecture import create_MDN_model, negative_loglikelihood

        input_len = reference_input_length(_resolution_of(name), _filter_of(name))
        model = create_MDN_model(input_len, negative_loglikelihood)
        model.load_weights(os.path.join(ML_DIR, name, name))
    else:
        predict_fn, input_len = _standard_reference(name)

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

    ref_out = predict_fn(X)
    ok = np.allclose(ref_out, onnx_out, rtol=RTOL, atol=ATOL)
    return ok, float(np.max(np.abs(ref_out - onnx_out)))


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
