# Ρομποτική Όραση, Διάλεξη 3: παραδείγματα για τις συναρτήσεις των διαφανειών.
# Κάθε κελί (# %%) τρέχει μόνο του. Φτιάχνουμε δική μας εικόνα.

# %% 0. Εισαγωγές, συνθετική εικόνα και ένα κομμάτι 3×3 που χρησιμοποιούν τα παραδείγματα
import cv2
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)          # σταθερή τυχαία γεννήτρια: ίδια αποτελέσματα κάθε φορά


def make_scene(h=120, w=160):
    """Φόντο 60, φωτεινό τετράγωνο 190 και θόρυβο. Επιστρέφει πίνακα uint8 (ύψος × πλάτος)."""
    img = np.full((h, w), 60, np.uint8)
    img[30:90, 50:110] = 190             # slicing: [γραμμές, στήλες]
    noisy = img + rng.normal(0, 8, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


gray = make_scene()
plt.imshow(gray, cmap="gray")
#plt.show()
print("σχήμα:", gray.shape, "| τύπος:", gray.dtype)

patch = np.array([[10, 10, 10],
                  [10, 90, 10],
                  [10, 10, 10]], dtype=np.uint8)   # το κομμάτι των διαφανειών (box και Gauss)

# %% 1. Ιστόγραμμα: plt.hist, cv2.calcHist, np.histogram
flat = gray.ravel()                      # ravel(): από πίνακα 2D σε λίστα τιμών 1D
print("πριν:", gray.shape, "| μετά το ravel:", flat.shape)

plt.hist(flat, bins=256, range=(0, 256))  # 256 κάδοι, εύρος τιμών (0, 256): tuple ως όρισμα
plt.title("Ιστόγραμμα")
plt.show()

hist_cv = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()   # λίστες: εικόνες, κανάλια
hist_np, _ = np.histogram(flat, bins=256, range=(0, 256))            # επιστρέφει tuple: (h, όρια)
print("calcHist = np.histogram:", np.array_equal(hist_cv.astype(int), hist_np))
print("το h(k) αθροίζει στο πλήθος των pixels:", hist_np.sum() == gray.size)

# %% 2. Εξισορρόπηση: cv2.equalizeHist και CLAHE (cv2.createCLAHE)
dark = (gray * 0.4).astype(np.uint8)     # σκοτεινή εκδοχή της εικόνας

equalized = cv2.equalizeHist(dark)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))   # τοπική εκδοχή: ανά πλακίδιο 8×8
local = clahe.apply(dark)

print("εύρος πριν    :", dark.min(), "-", dark.max())
print("equalizeHist  :", equalized.min(), "-", equalized.max())
print("CLAHE         :", local.min(), "-", local.max())

# Ο τύπος των διαφανειών «με το χέρι»: c(k) = αθροιστική κατανομή, g = round((L-1)·c)
h = np.bincount(dark.ravel(), minlength=256)
c = np.cumsum(h) / dark.size
lut = np.round(255 * c).astype(np.uint8)      # πίνακας αντιστοίχισης: τιμή -> νέα τιμή
manual = lut[dark]                            # fancy indexing: εφαρμογή σε όλα τα pixels
print("μέγιστη διαφορά από το equalizeHist:", np.abs(manual.astype(int) - equalized.astype(int)).max())

# %% 3. Συνέλιξη: cv2.filter2D με δικό μας πυρήνα
kernel = np.array([[1, 2, 1],
                   [2, 4, 2],
                   [1, 2, 1]], np.float32) / 16
out = cv2.filter2D(patch, -1, kernel)         # -1: ο τύπος εξόδου ίδιος με της εισόδου
print("pixel στο κέντρο μετά το φίλτρο:", out[1, 1], "(αναμενόμενο 30)")

# %% 4. Θόλωση με μέσο όρο: cv2.blur
print("box 3×3 στο κέντρο:", cv2.blur(patch, (3, 3))[1, 1], "(8·10 + 90) / 9 ≈ 19")   # (3, 3): tuple

# %% 5. Θόλωση Gaussian: cv2.GaussianBlur
print("Gaussian 3×3 στο κέντρο:", cv2.GaussianBlur(patch, (3, 3), 0)[1, 1], "(αναμενόμενο 30)")
# Το 0 στο τέλος σημαίνει «υπολόγισε το σ από το μέγεθος του πυρήνα».

# %% 6. Θόλωση median: cv2.medianBlur
noisy_patch = np.array([[10, 12, 255],
                        [11, 13, 14],
                        [9, 12, 10]], np.uint8)
print("median:", cv2.medianBlur(noisy_patch, 3)[1, 1], "| μέσος όρος:", cv2.blur(noisy_patch, (3, 3))[1, 1])
print("ταξινομημένες τιμές:", np.sort(noisy_patch.ravel()))      # η μεσαία (5η) είναι η διάμεσος

# %% 7. Ακμές με Sobel: cv2.Sobel, cv2.CV_64F και cv2.magnitude
edge_patch = np.array([[10, 10, 200]] * 3, np.uint8)             # λίστα × 3: τρεις ίδιες γραμμές
gx = cv2.Sobel(edge_patch, cv2.CV_64F, 1, 0, ksize=3)            # (1, 0): παράγωγος κατά x
gy = cv2.Sobel(edge_patch, cv2.CV_64F, 0, 1, ksize=3)            # (0, 1): παράγωγος κατά y
print("Gx =", gx[1, 1], "| Gy =", gy[1, 1], "| μέτρο =", cv2.magnitude(gx, gy)[1, 1])
# CV_64F: δεκαδικός τύπος, ώστε να χωράνε και αρνητικές τιμές (το uint8 θα τις έκοβε).

# %% 8. Ακμές με Canny: δύο κατώφλια
for low, high in [(50, 100), (100, 200), (150, 300)]:           # tuple unpacking μέσα στο for
    edges = cv2.Canny(gray, low, high)
    print(f"Canny({low}, {high}): {np.count_nonzero(edges)} pixels ακμής")

# %% 9. Κατωφλίωση: cv2.threshold (επιστρέφει tuple: κατώφλι, εικόνα)
values = np.array([[120, 130]], np.uint8)
t, binary = cv2.threshold(values, 127, 255, cv2.THRESH_BINARY)
print("κατώφλι:", t, "| αποτέλεσμα:", binary)                   # 120 -> 0, 130 -> 255

# %% 10. Otsu: το κατώφλι υπολογίζεται αυτόματα
t, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)   # σημαίες που προστίθενται
print("κατώφλι Otsu στη συνθετική σκηνή:", t)

# Ιστόγραμμα με δύο ίσες ομάδες (κορυφές στο 60 και στο 190), όπως στη διαφάνεια
two_groups = np.concatenate([rng.normal(60, 22, 5000), rng.normal(190, 25, 5000)])
two_groups = np.clip(two_groups, 0, 255).astype(np.uint8).reshape(100, 100)
t2, _ = cv2.threshold(two_groups, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print("κατώφλι Otsu με δύο ίσες ομάδες:", t2, "(περίπου στη μέση, ~125)")
# Όταν οι ομάδες έχουν πολύ διαφορετικό μέγεθος (όπως στη σκηνή), το κατώφλι μετακινείται.

# %% 11. Adaptive: σελίδα με σκιά (το φόντο σκουραίνει προς τα δεξιά)
x = np.linspace(0, 1, 200)
page = np.tile(220 - 120 * x, (100, 1))                          # np.tile: επανάληψη της γραμμής
for col in range(10, 200, 20):
    page[:, col:col + 3] -= 70                                   # σκούρες «γραμμές» (μελάνι)
page = page.astype(np.uint8)
plt.imshow(page,cmap="gray")
_, global_t = cv2.threshold(page, 127, 255, cv2.THRESH_BINARY)
plt.imshow(global_t,cmap="gray")

adaptive = cv2.adaptiveThreshold(page, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)       # γειτονιά 11×11, C = 2
plt.imshow(adaptive,cmap="gray")
print("μαύρα pixels, σταθερό κατώφλι:", round(np.mean(global_t == 0) * 100), "%")
print("μαύρα pixels, adaptive       :", round(np.mean(adaptive == 0) * 100), "% (μόνο το «μελάνι»)")

# %% 12. Μορφολογία: cv2.erode και cv2.dilate
mask = np.zeros((12, 16), np.uint8)
mask[2:10, 2:10] = 255                                           # αντικείμενο
mask[5, 5] = 0                                                   # μικρή τρύπα
mask[2, 13] = 255                                                # δύο μεμονωμένα pixels θορύβου
mask[7, 13] = 255
k = np.ones((3, 3), np.uint8)                                    # πυρήνας (πίνακας από άσους)

plt.imshow(mask,cmap="gray")
def white(m):
    return int(np.count_nonzero(m))


print("λευκά pixels: αρχικό =", white(mask),
      "| erode =", white(cv2.erode(mask, k, iterations=1)),
      "| dilate =", white(cv2.dilate(mask, k, iterations=1)))      # iterations: όρισμα με όνομα

# %% 13. Άνοιγμα και κλείσιμο: cv2.morphologyEx
opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)               # διάβρωση και μετά διαστολή
plt.imshow(opened,cmap="gray")
closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)              # διαστολή και μετά διάβρωση
plt.figure()
plt.imshow(closed,cmap="gray")
def objects(m):
    return cv2.connectedComponents(m)[0] - 1                    # -1: αφαιρούμε το φόντο


print("αντικείμενα: αρχικό =", objects(mask), "| άνοιγμα =", objects(opened), "| κλείσιμο =", objects(closed))
print("τρύπα στο (5, 5): άνοιγμα =", opened[5, 5], "| κλείσιμο =", closed[5, 5], "(255 = γεμάτο)")

# %% 14. Εργαστήριο: cv2.cvtColor (HSV) και cv2.inRange
red_bgr = np.uint8([[[0, 0, 255]]])                              # ένα pixel, σε σειρά B, G, R
hsv = cv2.cvtColor(red_bgr, cv2.COLOR_BGR2HSV)
print("κόκκινο σε HSV:", hsv[0, 0])                              # H = 0, S = 255, V = 255
mask_red = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
print("μέσα στα όρια του κόκκινου;", mask_red[0, 0] == 255)

# %%
