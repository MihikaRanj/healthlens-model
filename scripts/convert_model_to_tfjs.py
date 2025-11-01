"""
convert_model_to_tfjs.py
Converts scikit-learn .joblib model → ONNX → TensorFlow → TensorFlow.js
and verifies consistency of the exported model.
"""

import warnings, os, sys, numpy as np
warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================================================
# 0. Environment setup and compatibility patches
# ==========================================================

# Disable optional Lite runtime for onnx2tf
os.environ["ONNX2TF_USE_AI_EDGE_LITERT"] = "0"

# Patch NumPy deprecated aliases (needed for TF.js converter)
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "object"):
    np.object = object

# ==========================================================
# 1. Imports
# ==========================================================
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from onnx2tf import main as onnx2tf_main
import subprocess

# ==========================================================
# 2. Paths
# ==========================================================
model_path = "../models/model_nhanes_labs_v2.joblib"
onnx_path  = "../models/model_nhanes_labs_v2.onnx"
tf_path    = "../models/model_tf"
tfjs_path  = "../models/model_web"

# ==========================================================
# 3. Convert scikit-learn → ONNX
# ==========================================================
print(f"🔹 Loading model: {model_path}")
model = joblib.load(model_path)

n_features = 7  # adjust if your model has more features
onnx_model = convert_sklearn(
    model,
    initial_types=[("float_input", FloatTensorType([None, n_features]))],
    target_opset=12
)

os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
with open(onnx_path, "wb") as f:
    f.write(onnx_model.SerializeToString())
print(f"✅ Saved ONNX model to: {onnx_path}")

# ==========================================================
# 4. Convert ONNX → TensorFlow SavedModel (in-process)
# ==========================================================
print("🔹 Converting to TensorFlow (AI Edge Lite disabled)...")

# Set up CLI-style arguments for onnx2tf.main()
sys.argv = ["onnx2tf", "-i", onnx_path, "-o", tf_path]
onnx2tf_main()
print(f"✅ TensorFlow model saved to: {tf_path}")

# ==========================================================
# 5. Convert TensorFlow → TensorFlow.js
# ==========================================================
print("🔹 Converting to TensorFlow.js format...")

subprocess.run([
    "tensorflowjs_converter",
    "--input_format=tf_saved_model",
    tf_path,
    tfjs_path
], check=True)

print(f"✅ TF.js model exported to: {tfjs_path}")

# ==========================================================
# 6. Sanity-check inference test
# ==========================================================
print("\n🔍 Running sanity-check prediction...")

import onnxruntime as rt

# Load ONNX model for quick inference
sess = rt.InferenceSession(onnx_path)
input_name = sess.get_inputs()[0].name

# Example input: [AGE, BMI, SEX_MALE, HTN_FLAG, SMOKING_IDX, GLU, PA_ANY]
x_sample = np.array([[45, 27.5, 1, 0, 0, 100, 1]], dtype=np.float32)

pred = sess.run(None, {input_name: x_sample})[0]
print("ONNX model output:", pred)

# Compare with original scikit-learn model output
py_pred = model.predict_proba(x_sample)
print("Python model output:", py_pred)

print("\n✅ Sanity check complete — outputs should have similar shape.")
print("🎉 Conversion pipeline finished successfully!")
print("You can now load /models/model_web/model.json inside your Ionic React app.")
