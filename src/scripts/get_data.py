import torch
from torch import Tensor
from torchvision.utils import save_image
from jaxtyping import Float, UInt8
from einops import rearrange, repeat
from PIL import Image
from io import BytesIO
import torchvision.transforms as tf
to_tensor = tf.ToTensor()

def convert_poses(
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intrinsics

def convert_images(
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(to_tensor(image))
        return torch.stack(torch_images)


def get_data(data_path="/media/ubuntu/zoro/ubuntu/data/sbnset_nvs/re10k/re10k_subset/train/000000.torch", 
             idx1=0, idx2=10):

    chunk = torch.load(data_path)
    example = chunk[0]

    extrinsics, intrinsics = convert_poses(example["cameras"])
    num_scence_images = len(example["images"])
    idx2 = num_scence_images - 1

    img1 = example["images"][idx1]
    img2 = example["images"][idx2]
    imgs = convert_images([img1, img2])
    
    intri = intrinsics[idx1], intrinsics[idx2] # K  = 3X3
    extri = extrinsics[idx1], extrinsics[idx2] # RT = 4X4

    save_image(imgs[0], "./data/img1.png")  
    save_image(imgs[1], "./data/img2.png")

    return imgs, intri, extri 
   

if __name__ == "__main__":
    data_path = "/media/ubuntu/zoro/ubuntu/data/sbnset_nvs/re10k/re10k_subset/train/000000.torch"
    get_data(data_path)

