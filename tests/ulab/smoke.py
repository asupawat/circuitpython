try:
    import ulab as np
except ImportError:
    print("SKIP")
    raise SystemExit

np.array([1,2,3])
np.array([1,2,3], dtype=np.int8)
np.array([1,2,3], dtype=np.int16)
np.array([1,2,3], dtype=np.uint8)
np.array([1,2,3], dtype=np.uint16)
np.array([1,2,3], dtype=np.float)
np.zeros(3)
np.ones(3)
np.eye(3)
a = np.eye(3)
a.shape
a.size
a.itemsize
a.flatten
a.sort()
a.transpose()
a + 0
a + a
a * 0
a * a
a / 1
a / a
a - 0
a - a
+a
-a
a[0]
a[:]
a[0] = 0
a[:] = np.zeros((3,3))
a = np.eye(3)
np.acos(a)
np.acosh(a)
np.asin(a)
np.asinh(a)
np.atan(a)
np.atanh(a)
np.ceil(a)
np.cos(a)
#np.cosh(a)
np.sin(a)
np.sinh(a)
np.tan(a)
np.tanh(a)
np.erf(a)
np.erfc(a)
np.exp(a)
np.expm1(a)
np.floor(a)
np.gamma(a)
np.lgamma(a)
np.log(a)
np.log2(a)
np.sqrt(a)
np.dot(a,a)
np.inv(a)
np.eig(a)
np.det(a)
np.convolve(np.array([1,2,3]), np.array([1,10,100,1000]))
np.linspace(0, 10, num=3)
a = np.linspace(0, 10, num=256, endpoint=True)
np.spectrum(a)
p, q = np.fft(a)
np.ifft(p)
np.ifft(p,q)
np.argmin(a)
np.argmax(a)
np.argsort(a)
np.max(a)
np.min(a)
np.mean(a)
np.std(a)
np.diff(a)
np.size(a)
f = np.polyfit([1,2,3], 3)
np.polyval([1,2,3], [1,2,3])
np.sort(a)
np.flip(a)
np.roll(a, 1)
