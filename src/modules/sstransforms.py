import torch 
import numpy as np
def norm_grid(grid):
    B, H, W, C = grid.shape
    grid -= grid.view(-1, C).min(0).view(1, 1, 1, C)
    grid /= grid.view(-1, C).max(0).view(1, 1, 1, C)
    grid = (grid - 0.5) * 2
    return grid

def get_grid(shape, normalized=False):
    B, C, H, W = shape
    grid_x, grid_y = torch.meshgrid(torch.arange(H), torch.arange(W))
    grid_x = grid_x.float().cuda()
    grid_y = grid_y.float().cuda()
    indices = torch.stack([grid_y, grid_x], -1).view(1, H, W, 2).expand(B, H, W, 2).contiguous()
    if normalized:
        indices = norm_grid(indices)
    return indices

def get_elastic(grid, sigma, alpha):
    B, H, W, C = grid.shape
    sigma=self.exp_dict["model"]["sigma"]
    alpha=self.exp_dict["model"]["alpha"]
    dx = gaussian_filter((np.random.rand(B, H, W) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((np.random.rand(B, H, W) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dx = torch.from_numpy(dx).cuda().float()
    dy = torch.from_numpy(dy).cuda().float()
    dgrid_x = grid_x + dx
    dgrid_y = grid_y + dy
    dindices = torch.stack([dgrid_y, dgrid_x], -1).view(1, H, W, 2).expand(B, H, W, 2).contiguous()
    # grid = get_grid(images.shape)
    grid += dindices 
    grid = norm_grid(dindices)
    return dindices

def get_flip(grid, axis=1, random=True):
    if random:
        flips = torch.randint(low=0, high=2, size=(grid.size(0), 1, 1, 1), device=grid.device)
        flips *= 2
        flips -= 1
    else:
        flips = -1
    grid[..., axis] *= flips
    return grid 


def batch_rotation(grid, rots):
    ret = []
    for i, rot in enumerate(rots):
        ret.append(grid[i, ...].rot90(-int(rot // 90), [1,2]))
    return torch.stack(ret, 0)

def get_rotation(images, rot):
    if rot == 0:  # 0 degrees rotation
        return grid
    elif rot == 90:  # 90 degrees rotation
        return get_flip(grid.permute(0, 1, 3, 2), axis=0, random=False).contiguous()
    elif rot == 180:  # 90 degrees rotation
        return get_flip(get_flip(grid, 0, False), 1, False)
    elif rot == 270:  # 270 degrees rotation / or -90
        return get_flip(grid, 0, False).permute(0, 1, 3, 2).contiguous()
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')
# def get_rotation(grid, rot):
#     if rot == 0:  # 0 degrees rotation
#         return grid
#     elif rot == 90:  # 90 degrees rotation
#         return get_flip(grid.permute(0, 1, 3, 2), axis=0, random=False).contiguous()
#     elif rot == 180:  # 90 degrees rotation
#         return get_flip(get_flip(grid, 0, False), 1, False)
#     elif rot == 270:  # 270 degrees rotation / or -90
#         return get_flip(grid, 0, False).permute(0, 1, 3, 2).contiguous()
#     else:
#         raise ValueError('rotation should be 0, 90, 180, or 270 degrees')