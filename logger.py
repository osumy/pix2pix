# logger.py
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torchvision.utils as vutils

class Logger(object):
    def __init__(self, log_dir):
        """Create a SummaryWriter logging to log_dir."""
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        try:
            self.writer.add_scalar(tag, value, step)
        except Exception:
            pass

    def image_summary(self, tag, images, step):
        """Log images.
        - If `images` is a torch Tensor of shape (B,C,H,W), assumes values in [-1,1] or [0,1].
        - If `images` is a numpy HxWxC, or list of such, it will convert.
        """
        try:
            import torch
            # torch tensor (B,C,H,W)
            if torch.is_tensor(images):
                imgs = images.clone()
                # if in [-1,1]
                if imgs.min() < 0:
                    imgs = (imgs + 1.0) / 2.0
                grid = vutils.make_grid(imgs, nrow=min(8, imgs.size(0)), normalize=False)
                self.writer.add_image(tag, grid, step)
                return

            # numpy single image HWC
            if isinstance(images, np.ndarray):
                img = np.transpose(images, (2, 0, 1))
                self.writer.add_image(tag, img, step, dataformats='CHW')
                return

            # list of numpy images
            imgs = []
            for im in images:
                if isinstance(im, np.ndarray):
                    imgs.append(np.transpose(im, (2, 0, 1)))
            if len(imgs) > 0:
                import torch
                imgs_t = torch.from_numpy(np.stack(imgs)).float()
                if imgs_t.max() > 2:
                    imgs_t = imgs_t / 255.0
                grid = vutils.make_grid(imgs_t, nrow=min(8, imgs_t.size(0)), normalize=False)
                self.writer.add_image(tag, grid, step)
        except Exception:
            pass

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""
        try:
            self.writer.add_histogram(tag, values, step, bins=bins)
        except Exception:
            pass

    def close(self):
        try:
            self.writer.close()
        except Exception:
            pass
