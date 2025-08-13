## Prerequisites

1. Install required libraries

   ```python
   pip install -r requirements.txt
   ```

   

2. Install pytorch-3dunet from pytorch-3dunet-master.zip



## Train

- Data Preprocess

  Generate mask images from labeled swc files.

  ```
  python PreMakeData.py
  ```

  Cut raw images and mask images into appropriate size with command

  ```
  cd ./preprocess
  Cut_data.exe
  ```

  - Input : Raw images of size $512\times 512 \times 512$ and corresponding labeled swc files.
  - Output : Cutted raw images of size $128\times 128 \times 128$ and corresponding mask images.

- Train Segment model with command

  ``` python
  python Train.py
  ```

  

## Run

- Run model with command

  ```
  python BigImgPredict.py
  ```

  Input : Raw images of size  $512\times 512 \times 512$

  Output : Segmentation images of size $512\times 512 \times 512$

## Citation

If you find our work useful in your research, please consider citing our paper:

```
Cai Lin, Fan Taiyu, Qu Xuzhong, Zhang Ying, Gou Xianyu, Ding Quanwei, Feng Weihua, Cao Tingting, Lv Xiaohua, Liu Xiuli, Huang Qing, Quan Tingwei, Zeng Shaoqun (2024) PointTree: Automatic and accurate reconstruction of long-range axonal projections of single-neuron eLife 13:RP102840

https://doi.org/10.7554/eLife.102840.2
```

If you have questions, feel free to contact cailin0227@hust.edu.cn













