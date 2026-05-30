# Locally-built CUDA wheels for the Jetson

`pyproject.toml`'s `[tool.uv.sources]` block pins `ctranslate2` to a wheel in
this directory **on aarch64 only** (i.e. the Jetson Orin Nano). On Mac the
marker is false and uv resolves to PyPI's CPU wheel as usual.

The wheels themselves are gitignored — they're machine- and CUDA-version-
specific. To rebuild on the Jetson:

```bash
sudo apt install -y cmake build-essential libcudnn9-dev libcudnn9-cuda-12

cd ~
git clone --recursive --branch v4.7.2 https://github.com/OpenNMT/CTranslate2.git
cd CTranslate2 && mkdir build && cd build
cmake .. \
  -DWITH_CUDA=ON -DWITH_CUDNN=ON -DWITH_MKL=OFF \
  -DOPENMP_RUNTIME=COMP \
  -DCUDA_ARCH_LIST="8.7" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTS=OFF -DBUILD_CLI=OFF
make -j$(nproc)        # 30–60 min on Orin Nano
sudo make install
sudo ldconfig

cd ../python
# IMPORTANT: use the brain venv's Python (3.12), not the system one.
uv pip install --python ~/code/stackchanagent/brain/.venv/bin/python \
  -r install_requirements.txt
~/code/stackchanagent/brain/.venv/bin/python setup.py bdist_wheel

# Copy into this directory; the pyproject.toml source pin picks it up.
cp dist/ctranslate2-4.7.2-cp312-cp312-linux_aarch64.whl \
   ~/code/stackchanagent/brain/wheels/
```

After dropping a new wheel in, `uv sync` will install it on the next run.
