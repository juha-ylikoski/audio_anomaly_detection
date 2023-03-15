
import numpy as np
import imageio.v3 as iio
from simplejpeg import decode_jpeg, encode_jpeg




def make_bad(path, quality):
    im = iio.imread(path)
    compressed = encode_jpeg(im, quality=quality)
    print(type(compressed))
    bad = decode_jpeg(compressed)
    return bad, compressed


def psnr(a: np.array, b: np.array, max_val: int = 255) -> float:
    return 20 * np.log10(max_val) - 10 * np.log10(np.power((a - b), 2).mean())


bad, compressed = make_bad("/tmp/test/test.png", 0)
orig = iio.imread("/tmp/test/test.png")
iio.imwrite("bad.jpg", bad)
print(psnr(orig, bad))

num_pixels = 512*512

bpp = len(compressed) * 8.0 / num_pixels

print(bpp)

