import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from skimage import feature
import scipy.ndimage as ndi
from scipy.ndimage._filters import _gaussian_kernel1d
from skimage._shared.filters import gaussian as _sk_gaussian
from skimage._shared.utils import _supported_float_type
from skimage.feature._canny_cy import _nonmaximum_suppression_bilinear


class _Canny:
    """skimage.feature.canny(image, sigma) with the per-shape constants cached.

    Bit-identical to skimage.feature.canny with default thresholds, mask=None and
    mode='constant' for float32/float64 images: the same scipy/skimage primitives
    are called in the same order; only the bleed-over normalisation (gaussian of a
    ones mask) and the eroded border mask, which depend on the shape alone, are
    computed once instead of on every call.
    """

    def __init__(self, sigma):
        self.sigma = sigma
        self._const = {}
        # gaussian_filter1d kernel (radius int(4 * sigma + 0.5)), reversed for correlate1d
        self._weights = np.ascontiguousarray(
            _gaussian_kernel1d(float(sigma), 0, int(4.0 * float(sigma) + 0.5))[::-1], dtype=np.float64)

    def _constants(self, shape, dtype):
        key = (shape, dtype)
        c = self._const.get(key)
        if c is None:
            float_type = _supported_float_type(dtype)
            mask = np.ones(shape, dtype=float_type)
            bleed_over = (_sk_gaussian(mask, sigma=self.sigma, mode='constant', cval=0.0, preserve_range=False)
                          + np.finfo(float_type).eps)
            eroded_mask = np.ones(shape, dtype=bool)
            eroded_mask[:1, :] = 0
            eroded_mask[-1:, :] = 0
            eroded_mask[:, :1] = 0
            eroded_mask[:, -1:] = 0
            c = (bleed_over, eroded_mask)
            if not self._self_check(shape, dtype, c):
                # the cached path relies on private skimage/scipy helpers; if an
                # installed version makes it disagree with skimage, use skimage itself
                print("warning: fast canny differs from skimage.feature.canny for %s %s, "
                      "falling back to skimage" % (shape, np.dtype(dtype).name))
                c = False
            self._const[key] = c
        return c

    def _self_check(self, shape, dtype, consts, n=8):
        # compare against skimage on deterministic images in the screener's value range
        # (grayscale of a [0, 255] image): pure noise, smooth gradients and blocky edges
        rng = np.random.default_rng(0)
        yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
        for i in range(n):
            if i % 3 == 0:
                img = rng.uniform(0, 255, shape)
            elif i % 3 == 1:
                img = 127.5 + 127.5 * np.sin(xx * rng.uniform(0.1, 1) + yy * rng.uniform(0.1, 1))
            else:
                img = np.kron(rng.uniform(0, 255, (shape[0] // 4 + 1, shape[1] // 4 + 1)),
                              np.ones((4, 4)))[:shape[0], :shape[1]]
            img = img.astype(dtype)
            try:
                if not np.array_equal(self._fast(img, *consts), feature.canny(img, sigma=self.sigma)):
                    return False
            except Exception:
                return False
        return True

    def __call__(self, image):
        if image.ndim != 2 or image.dtype not in (np.float32, np.float64):
            return feature.canny(image, sigma=self.sigma)
        c = self._constants(image.shape, image.dtype)
        if c is False:
            return feature.canny(image, sigma=self.sigma)
        return self._fast(image, *c)

    def _fast(self, image, bleed_over, eroded_mask):
        # skimage gaussian(..., preserve_range=False) on a float image is
        # ndi.gaussian_filter(image, sigma, mode='constant', cval=0, truncate=4), i.e.
        # the same correlate1d calls as below with the (cached) gaussian kernel
        w = self._weights
        smoothed = np.zeros(image.shape, dtype=image.dtype)
        ndi.correlate1d(image, w, 0, smoothed, 'constant', 0.0, 0)
        ndi.correlate1d(smoothed, w, 1, smoothed, 'constant', 0.0, 0)
        smoothed /= bleed_over

        # ndi.sobel(smoothed, axis=1) and ndi.sobel(smoothed, axis=0), mode='reflect'
        jsobel = np.zeros(image.shape, dtype=image.dtype)
        ndi.correlate1d(smoothed, _D, 1, jsobel, 'reflect', 0.0, 0)
        ndi.correlate1d(jsobel, _S, 0, jsobel, 'reflect', 0.0, 0)
        isobel = np.zeros(image.shape, dtype=image.dtype)
        ndi.correlate1d(smoothed, _D, 0, isobel, 'reflect', 0.0, 0)
        ndi.correlate1d(isobel, _S, 1, isobel, 'reflect', 0.0, 0)
        magnitude = isobel * isobel
        magnitude += jsobel * jsobel
        np.sqrt(magnitude, out=magnitude)

        low_masked = _nonmaximum_suppression_bilinear(isobel, jsobel, magnitude, eroded_mask, 0.1)
        low_mask = low_masked > 0
        labels, count = ndi.label(low_mask, _STREL)
        if count == 0:
            return low_mask
        high_mask = low_mask & (low_masked >= 0.2)
        nonzero_sums = np.unique(labels[high_mask])
        good_label = np.zeros((count + 1,), bool)
        good_label[nonzero_sums] = True
        return good_label[labels]


_STREL = np.ones((3, 3), bool)
_D = np.array([-1, 0, 1], dtype=np.float64)
_S = np.array([1, 2, 1], dtype=np.float64)
_EPS = np.float32(0.00000000001)


class Screener:
    def __init__(self, enable=False, thold=0.3, sigma=3, n=10, dim=40):
        self.enable = enable
        self.thold = thold
        self.sigma = sigma # edge detail
        self.n = n # fifo depth
        self.dimension = dim # square dimension
        assert dim % 8 == 0, "dimension must be multiple of 8"
        self.lut = []
        for i in range(8):
            byte = 1 << 7-i
            self.lut.append(byte)
        self._canny = _Canny(sigma)
        # FIFO of edge maps kept as a ring buffer of bit-packed rows (uint64 words)
        # plus the per-row edge count. The order of rows does not matter for the
        # min-distance screen, only which n edge maps are in the window.
        self._words = dim * dim // 64
        self._buf = np.zeros((n, self._words), dtype=np.uint64)
        self._cnt = np.zeros(n, dtype=np.int64)
        self._len = 0
        self._pos = 0

    @property
    def fifo(self):
        # edge maps currently in the window, oldest first (for inspection only)
        idx = [(self._pos - self._len + i) % self.n for i in range(self._len)]
        bits = np.unpackbits(self._buf[idx].view(np.uint8), axis=1).astype(bool)
        return [torch.from_numpy(b) for b in bits]

    def reset(self):
        self._len = 0
        self._pos = 0
        return

    def _edges(self, image):
        # flat boolean numpy edge map (internal fast path)
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        image = image.squeeze()

        # image = ndi.gaussian_filter(image, 1)
        return self._canny(image).ravel()

    def gen_edge(self, x):
        # public API kept as before: flat bool torch tensor
        return torch.from_numpy(self._edges(x))

    def down_size(self, e):
        xe = torch.zeros(int(self.dimension*self.dimension/8))
        idx = 0
        byte = 0
        for i in range(len(e)):
            if i > 0 and i % 8 == 0:
                xe[idx] = byte
                idx += 1
                byte = 0
            if e[i]:
                byte = byte | self.lut[i % 8]
        return xe

    def input_transform(self, x):
        # public API kept as before: flat bool torch tensor edge map (zero-copy view)
        return torch.from_numpy(self._edge_map(x))

    def _edge_map(self, x):
        # x: 3xHxW (or 1x3xHxW) image, torch tensor or numpy array.
        # Same arithmetic as T.Grayscale on a tensor, (0.2989*r + 0.587*g + 0.114*b)
        # in the input dtype; T.Resize to the size the image already has is the identity.
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = x.reshape(x.shape[-3:])
        r, g, b = x[0], x[1], x[2]
        dt = x.dtype.type
        gray = dt(0.2989) * r + dt(0.587) * g + dt(0.114) * b
        if gray.shape != (self.dimension, self.dimension):
            t = torch.from_numpy(np.ascontiguousarray(gray))[None]
            gray = T.Resize((self.dimension, self.dimension))(t)[0].numpy()

        xe = self._edges(gray)
        #xe = self.down_size(e)
        return xe

    def screen(self, e0, e1):
        # bitwise and to get difference
        d = np.sum(e1 ^ e0)
        # rate of difference
        s = np.sum(e0) + np.sum(e1)
        v = d/(s + 0.00000000001)

        # threshold
        flag = False
        if v < self.thold:
            flag = True
        return flag

    def fifo_full(self):
        if self._len == self.n:
            return True
        return False

    def fifo_in(self, packed, count):
        # if the fifo is full the oldest slot is overwritten by the new example
        self._buf[self._pos] = packed
        self._cnt[self._pos] = count
        self._pos = (self._pos + 1) % self.n
        if self._len < self.n:
            self._len += 1
        return

    def run(self, x):
        # pre-process to screen out dummy injections
        suspicious = True

        if not self.enable:
            # pre-process is disabled, always return true to pass example to gwad
            return True

        # transform example
        xn = self._edge_map(x)
        packed = np.packbits(xn).view(np.uint64)
        cnt = int(np.count_nonzero(xn))
        k = self._len
        if k == 0:
            # first query is always screened out
            suspicious = False
        else:
            # same arithmetic as the torch version: integer counts, then the
            # float32 d / (s + 1e-11) and its min over the window
            d = np.bitwise_count(self._buf[:k] ^ packed).sum(axis=1)
            s = (self._cnt[:k] + cnt).astype(np.float32) + _EPS
            dist = d.astype(np.float32) / s
            score = dist.min()
            if score >= self.thold:
                # no suspicious query exists in the window, so dummy to be screened out
                suspicious = False
        self.fifo_in(packed, cnt)
        return suspicious


class ExampleSave:
    def __init__(self, mu, std, format):
        # target model training dataset setting
        self.data_mu = mu
        self.data_std = std
        self.data_foramt = format

    def ready_img(self, x, format=None):
        # format: 'tensor' -> CHW float tensor in [0, 255], otherwise HWC uint8 numpy (RGB)
        # The per-channel x*std + mu is elementwise in the input dtype, so doing it
        # after the single device->host copy gives the same values as on the GPU.
        format = self.data_foramt if format is None else format
        shape = x.size()
        img = x.detach().reshape(shape[1], shape[2], shape[3]).cpu().numpy()
        dt = img.dtype.type
        img = img * np.array(self.data_std, dtype=dt)[:, None, None] + np.array(self.data_mu, dtype=dt)[:, None, None]

        if format == 'tensor':
            img *= dt(255)
            img = torch.from_numpy(img)
        else:
            img = img.transpose(1, 2, 0)
            img = np.clip(img * 255, 0, 255)
            img = np.uint8(img)
        return img

    def imwrite(self, x, directory, id):
        img = self.ready_img(x, format='numpy')

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # assuming the model is pretrained with RGB images.
        os.makedirs(directory, exist_ok=True)
        cv2.imwrite(os.path.join(directory, '{}.jpg'.format(id)), img)
        return

    def sequence(self, x, id):
        directory = 'tmp/sequence/'
        self.imwrite(x, directory, id)
        return

    def attack_screened(self, x, id):
        directory = 'tmp/attack_screened/'
        self.imwrite(x, directory, id)
        return

    def attack_passed(self, x, id):
        directory = 'tmp/attack_passed/'
        self.imwrite(x, directory, id)
        return

    def dummy_screened(self, x, id):
        directory = 'tmp/dummy_screened/'
        self.imwrite(x, directory, id)
        return

    def dummy_passed(self, x, id):
        directory = 'tmp/dummy_passed/'
        self.imwrite(x, directory, id)
        return