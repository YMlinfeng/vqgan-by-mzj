# VQGAN
##### CVPR 2021 (Oral)
![teaser](assets/mountain.jpeg)

[**Taming Transformers for High-Resolution Image Synthesis**](https://compvis.github.io/taming-transformers/)<br/>
[Patrick Esser](https://github.com/pesser)\*,
[Robin Rombach](https://github.com/rromb)\*,
[Björn Ommer](https://hci.iwr.uni-heidelberg.de/Staff/bommer)<br/>
\* equal contribution

**tl;dr** We combine the efficiancy of convolutional approaches with the expressivity of transformers by introducing a convolutional VQGAN, which learns a codebook of context-rich visual parts, whose composition is modeled with an autoregressive transformer.

![teaser](assets/teaser.png)
[arXiv](https://arxiv.org/abs/2012.09841) | [BibTeX](#bibtex) | [Project Page](https://compvis.github.io/taming-transformers/)


## Requirements
A suitable [conda](https://conda.io/) environment named `taming` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate taming
```
## Overview of pretrained models
The following table provides an overview of all models that are currently available. 
FID scores were evaluated using [torch-fidelity](https://github.com/toshas/torch-fidelity).
For reference, we also include a link to the recently released autoencoder of the [DALL-E](https://github.com/openai/DALL-E) model. 
See the corresponding [colab
notebook](https://colab.research.google.com/github/CompVis/taming-transformers/blob/master/scripts/reconstruction_usage.ipynb)
for a comparison and discussion of reconstruction capabilities.

| Dataset  | FID vs train | FID vs val | Link |  Samples (256x256) | Comments
| ------------- | ------------- | ------------- |-------------  | -------------  |-------------  |
| FFHQ (f=16) | 9.6 | -- | [ffhq_transformer](https://k00.fr/yndvfu95) |  [ffhq_samples](https://k00.fr/j626x093) |
| CelebA-HQ (f=16) | 10.2 | -- | [celebahq_transformer](https://k00.fr/2xkmielf) | [celebahq_samples](https://k00.fr/j626x093) |
| ADE20K (f=16) | -- | 35.5 | [ade20k_transformer](https://k00.fr/ot46cksa) | [ade20k_samples.zip](https://heibox.uni-heidelberg.de/f/70bb78cbaf844501b8fb/) [2k] | evaluated on val split (2k images)
| COCO-Stuff (f=16) | -- | 20.4  | [coco_transformer](https://k00.fr/2zz6i2ce) | [coco_samples.zip](https://heibox.uni-heidelberg.de/f/a395a9be612f4a7a8054/) [5k] | evaluated on val split (5k images)
| ImageNet (cIN) (f=16) | 15.98/15.78/6.59/5.88/5.20 | -- | [cin_transformer](https://k00.fr/s511rwcv) | [cin_samples](https://k00.fr/j626x093) | different decoding hyperparameters |  
| |  | | || |
| FacesHQ (f=16) | -- |  -- | [faceshq_transformer](https://k00.fr/qqfl2do8)
| S-FLCKR (f=16) | -- | -- | [sflckr](https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/) 
| D-RIN (f=16) | -- | -- | [drin_transformer](https://k00.fr/39jcugc5)
| | |  | | || |
| VQGAN ImageNet (f=16), 1024 |  10.54 | 7.94 | [vqgan_imagenet_f16_1024](https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/) | [reconstructions](https://k00.fr/j626x093) | Reconstruction-FIDs.
| VQGAN ImageNet (f=16), 16384 | 7.41 | 4.98 |[vqgan_imagenet_f16_16384](https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/)  |  [reconstructions](https://k00.fr/j626x093) | Reconstruction-FIDs.
| VQGAN OpenImages (f=8), 256 | -- | 1.49 |https://ommer-lab.com/files/latent-diffusion/vq-f8-n256.zip |  ---  | Reconstruction-FIDs. Available via [latent diffusion](https://github.com/CompVis/latent-diffusion).
| VQGAN OpenImages (f=8), 16384 | -- | 1.14 |https://ommer-lab.com/files/latent-diffusion/vq-f8.zip  |  ---  | Reconstruction-FIDs. Available via [latent diffusion](https://github.com/CompVis/latent-diffusion)
| VQGAN OpenImages (f=8), 8192, GumbelQuantization | 3.24 | 1.49 |[vqgan_gumbel_f8](https://heibox.uni-heidelberg.de/d/2e5662443a6b4307b470/)  |  ---  | Reconstruction-FIDs.
| | |  | | || |
| DALL-E dVAE (f=8), 8192, GumbelQuantization | 33.88 | 32.01 | https://github.com/openai/DALL-E | [reconstructions](https://k00.fr/j626x093) | Reconstruction-FIDs.



## Data Preparation

### ImageNet
The code will try to download (through [Academic
Torrents](http://academictorrents.com/)) and prepare ImageNet the first time it
is used. However, since ImageNet is quite large, this requires a lot of disk
space and time. If you already have ImageNet on your disk, you can speed things
up by putting the data into
`${XDG_CACHE}/autoencoders/data/ILSVRC2012_{split}/data/` (which defaults to
`~/.cache/autoencoders/data/ILSVRC2012_{split}/data/`), where `{split}` is one
of `train`/`validation`. It should have the following structure:

```
${XDG_CACHE}/autoencoders/data/ILSVRC2012_{split}/data/
├── n01440764
│   ├── n01440764_10026.JPEG
│   ├── n01440764_10027.JPEG
│   ├── ...
├── n01443537
│   ├── n01443537_10007.JPEG
│   ├── n01443537_10014.JPEG
│   ├── ...
├── ...
```





### Other
- A [video summary](https://www.youtube.com/watch?v=o7dqGcLDf0A&feature=emb_imp_woyt) by [Two Minute Papers](https://www.youtube.com/channel/UCbfYPyITQ-7l4upoX8nvctg).
- A [video summary](https://www.youtube.com/watch?v=-wDSDtIAyWQ) by [Gradient Dude](https://www.youtube.com/c/GradientDude/about).


![txt2img](assets/birddrawnbyachild.png)

Text prompt: *'A bird drawn by a child'*

## 发布 Python 包到 PyPI 的命令汇总
```
# 1. 安装打包上传工具（setuptools、wheel、twine）
pip install setuptools wheel twine

# 2. 升级 pip（可选但推荐）
python3 -m pip install --upgrade pip

# 3. 为避免打包失败，升级 setuptools、packaging、wheel（确保兼容）
python3 -m pip install --upgrade setuptools packaging wheel

# 4. 检查 setup.py 文件编码错误（添加编码声明，以防中文注释报错）
# 在 setup.py 顶部加上：
# -*- coding: utf-8 -*-

# 5. 构建分发文件（源码包 + wheel 包）
python3 setup.py sdist bdist_wheel

# 6. 上传到 PyPI（会提示输入 API token，格式类似 pypi-xxxxxxxx...）
twine upload dist/*

# 上传成功后可以访问你的包页面：
# https://pypi.org/project/vqgan-by-mzj/

# ✅ 可选：如果使用 test.pypi.org 测试上传
# twine upload --repository-url https://test.pypi.org/legacy/ dist/*

```

---

## ✅ `pip install .` 和 `pip install -e .` 的区别

| 命令 | 含义 | 安装行为 | 适用场景 |
|------|------|-----------|-----------|
| `pip install .` | 安装当前目录下的包 | ✅ **复制包源码**到 `site-packages` 目录 | 正式安装、发布测试 |
| `pip install -e .` | 安装当前目录的包为“可编辑模式” | 🔗 在 `site-packages` 创建一个 **指向当前目录的符号链接**，源码不复制 | 开发调试、频繁改代码 |

---

## 🧪 举例说明

假设你当前目录下有如下结构：

```
my_project/
├── setup.py
└── my_package/
    └── __init__.py
```

### ▶ 使用 `pip install .`

执行后：

- `my_package/` 会被**复制**到 `site-packages`
- 你可以在任何地方 `import my_package`
- **但是**你修改 `my_project/my_package/` 里的源码，不会影响实际安装的包，除非你重新运行 `pip install .`

---

### ▶ 使用 `pip install -e .`

执行后：

- `site-packages` 下不会复制代码，而是创建 `.egg-link` 文件，指向你的项目目录
- 所以你改动 `my_package/` 中的代码，**立即生效**，无需重新安装
- 非常适合开发和调试

可以用这个命令查看链接路径：

```bash
python -m site
```

---

## 🔁 变化对比总结

| 项目变动 | `pip install .` 反应 | `pip install -e .` 反应 |
|-----------|--------------------|-------------------------|
| 修改 Python 代码 | ❌ 不会生效，除非重新安装 | ✅ 立刻生效 |
| 添加新模块 | ❌ 不生效 | ✅ 生效（只要 `setup.py` 中包含） |
| 删除模块 | ❌ 不生效 | ✅ 生效 |
| 删除项目目录 | ✅ 不影响 `site-packages` 已安装版本 | ❌ 导致导入失败（因为是软链接） |

---

## 📌 总结一句话：

> **`pip install .` 是常规安装，复制代码；`pip install -e .` 是开发模式安装，软链接源码。**

---

如果你正在开发自己的包（比如 `vqgan-by-mzj`），强烈推荐使用：

```bash
pip install -e .
```

这样每次改完代码就能直接测试，无需反复打包安装。




## git 代码

### 标准版

```
cd vqgan-by-mzj
git init
git status
git add .
git commit -m "Initial commit"
git push -u origin main
```

### 其他操作
- `git remote set-url origin https://github.com/YMlinfeng/vqgan-by-mzj.git`
- git remote -v
- git branch
- git branch -m master main 
- git pull origin main --allow-unrelated-histories (拉取远程分支并强制合并)
