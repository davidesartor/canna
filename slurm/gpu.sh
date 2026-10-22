# Sourced by the GPU jobs, after slurm/run-src.sh when there is one (it sets the run's LISA_BF16).
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# CUDA-graph capture of the cuDNN attention kernel fails on these nodes (cudaGraphAddMemsetNode)
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
# a float32 backbone compiles wrong under XLA's GEMM autotuning, and TF32 is not float32 (F20, F22, F23)
if [[ "${LISA_BF16:-backbone}" != backbone ]]; then
    XLA_FLAGS+=" --xla_gpu_autotune_level=0"
    export JAX_DEFAULT_MATMUL_PRECISION=highest
fi
# uv trusts the system store, which carries the CNES egress proxy's CA (UV_NATIVE_TLS on uv < 0.12)
export UV_SYSTEM_CERTS=1
echo "host      : $(hostname)"
echo "job       : ${SLURM_JOB_ID:-none}"
echo "date      : $(date)"
echo "XLA_FLAGS : ${XLA_FLAGS}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
