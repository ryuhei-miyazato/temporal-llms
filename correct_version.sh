# 1) conda を有効化して対象 env へ
source ~/miniconda3/etc/profile.d/conda.sh
conda activate temporal_llms

# 2) まず pip 由来の衝突パッケージを掃除
python -m pip uninstall -y torch torchvision torchaudio \
  numpy pandas scipy scikit-learn pyarrow || true

# 3) conda で土台（GPU対応 PyTorch + 数値系）を固定
#   - CUDA 12.1 ランタイム同梱（ドライバは 535+ 目安）
conda install -y -c pytorch -c nvidia pytorch=2.3.1 pytorch-cuda=12.1

# NumPy は 1.26 系、pandas/scipy/sklearn/pyarrow を互換組合せで
conda install -y -c conda-forge \
  numpy=1.26.4 pandas=2.2.2 scipy=1.11.4 scikit-learn=1.3.2 pyarrow=12

# 4) HF スタックを更新（Transformers 4.44+ 推奨）
python -m pip install -U \
  "transformers>=4.44" "tokenizers>=0.19" "accelerate>=0.30" \
  "huggingface_hub>=0.24" "safetensors>=0.4"

# 5) スモークテスト
python - <<'PY'
import torch, numpy, pandas
print("torch:", torch.__version__, "cuda?", torch.cuda.is_available())
print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
from transformers import AutoTokenizer, AutoModelForCausalLM
name="meta-llama/Llama-3.1-8B-Instruct"
tok=AutoTokenizer.from_pretrained(name)
m=AutoModelForCausalLM.from_pretrained(name, device_map="auto", torch_dtype=torch.float16)
print("loaded:", m.config.model_type, "dtype:", m.dtype)
PY
