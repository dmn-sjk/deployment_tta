# TensorRT 10.3, CUDA 12.6, cuDNN 9.3, Python 3.10: matches JetPack 6.2. Do not bump without checking TensorRT parity.
FROM nvcr.io/nvidia/pytorch:24.08-py3

# CPU reference backend for Polygraphy ONNX vs TensorRT comparisons.
RUN pip install --no-cache-dir onnxruntime==1.19.2

# Experiment files. The base image ships this version: pinned so it cannot drift.
RUN pip install --no-cache-dir pyyaml==6.0.2

# Formatters, linter and type checker. Config in pyproject.toml. tomli: docformatter reads it on Python 3.10.
RUN pip install --isolated --no-cache-dir ruff==0.17.1 mypy==2.4.0 "docformatter[tomli]==1.7.8"

# TensorRT ships no type stubs. Generate them from the pybind11 bindings as a PEP 561 tensorrt-stubs package for Pylance; mypy skips them
# (pyproject.toml). Plugin methods name a parameter `in`, a keyword, so rename it for the stubs to parse.
RUN pip install --no-cache-dir pybind11-stubgen==2.5.1 \
 && pybind11-stubgen tensorrt -o /tmp/stubs \
 && sed -i -E 's/([(,] )in: /\1in_: /' /tmp/stubs/tensorrt/*.pyi \
 && mv /tmp/stubs/tensorrt /usr/local/lib/python3.10/dist-packages/tensorrt-stubs \
 && rm -rf /tmp/stubs

ENV PYTHONPATH=/workspace/deployment_tta/src \
    PYTHONDONTWRITEBYTECODE=1

# NGC sets this to 1, which forces TF32 in cuBLAS and breaks the FP32 reference.
ENV TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
# Required by torch.use_deterministic_algorithms(True) for cuBLAS.
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
