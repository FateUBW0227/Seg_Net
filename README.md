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

  Output : Segment images of size $512\times 512 \times 512$











