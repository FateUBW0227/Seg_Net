## Prerequisites

1. Install required library

   ```python
   pip install -r requirements.txt
   ```

   

2. Install pytorch-3dunet from https://github.com/wolny/pytorch-3dunet



## Train

- Data Preprocess

  ```
  
  ```

  - Input : Raw images of size $512\times 512 \times 512$ and corresponding labeled swc files.
  - Output : Cutted raw image of size $128\times 128 \times 128$ and corresponding mask images.

- Train Segment model with command

  ``` python
  python Train.py
  ```

  

## Run

- Run model with command

  ```
  python BigImgPredict.py
  ```

  











