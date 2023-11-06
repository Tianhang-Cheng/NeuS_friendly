# A custom-data friendly [NeuS](https://github.com/Totoro97/NeuS)

This project is forked from NeuS but converted from NeuS space to the more common NeRF space (or OpenCV space).
It also fixes some bugs I encountered and adds a more detailed guidance on how to use colmap to build your own NeuS datasets.

## Setup

Clone this repository

```shell
git clone https://github.com/Totoro97/NeuS.git
cd NeuS
pip install -r requirements.txt
```

## extract colmap pose from custom multi-view image

In the first time runining, this script will generate a .xyz file and you can use MeshLab to edit the interesting area.
Save the .xyz inplace and run this script again.
Remeber to modify the path in the script.

```bash
python process_custom_data.py
```

## training

```bash
python exp_runner.py --mode train --conf ./confs/custom_colmap_data_womask.conf --case hotdog
```

## bugs of original NeuS:

1. (on windows) pyparsing.exceptions.ParseSyntaxException: , found '='  (at char 872), (line:50, col:14)

Solution: delete these content in .conf file:
    "d_in_view = 3" and "d_in" of "nerf"
    "d_in" and "d_out" of "sdf_network"
    "d_out" and "d_in" of "rendering_network"

2. Device mismatch

torch.randint put data in cuda by default in my envrionment

```bash
  File "E:\code\NeuS\models\dataset.py", line 118, in gen_random_rays_at
    color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (cpu)
```

Solution: manipulate deive by hand for torch.zeros(), torch.randn(), torch.ones() etc.

```python
pixels_x = torch.randint(low=0, high=self.W, size=[batch_size]).cpu()
pixels_y = torch.randint(low=0, high=self.H, size=[batch_size]).cpu()
color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
p = torch.stack([pixels_x.cuda(), pixels_y.cuda(), torch.ones_like(pixels_y).cuda()], dim=-1).float()  # batch_size, 3
```

3. Unexpected behavior of "load_K_Rt_from_P"

the result shows that the "load_K_Rt_from_P" may produce inconsistent result because it use

```python

def load_K_Rt_from_P(P): 
    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32) 
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

intrinsics
>>array([[1090.26322186,    0.        ,  400.        ,    0.        ],
          [   0.        , 1090.17241294,  400.        ,    0.        ],
          [   0.        ,    0.        ,    1.        ,    0.        ],
          [   0.        ,    0.        ,    0.        ,    1.        ]])

w2c
>>array([[-0.41983511, -0.86389187,  0.27826123, -0.43694697],
          [ 0.89267481, -0.33767177,  0.29851206, -0.4722799 ],
          [-0.16392118,  0.37372263,  0.91294098,  1.70342235],
          [ 0.        ,  0.        ,  0.        ,  1.        ]])
intrinsics_test, w2c_test = load_K_Rt_from_P(intrinsics @ w2c)

w2c_test
>>array([[-0.41983512, -0.8638919 ,  0.2782612 ,  0.5173737 ],
        [ 0.8926748 , -0.3376718 ,  0.29851207, -1.173558  ],
        [-0.16392118,  0.37372264,  0.912941  , -1.2925575 ],
        [ 0.        ,  0.        ,  0.        ,  1.        ]],
        dtype=float32)

intrinsics_test
>>array([[1090.26322186,    0.        ,  400.        ,    0.        ],
        [   0.        , 1090.17241294,  400.        ,    0.        ],
        [   0.        ,    0.        ,    1.        ,    0.        ],
        [   0.        ,    0.        ,    0.        ,    1.        ]])
```

We can find that the "w2c_test" is not equal to the input "w2c"

Solution: remove this function