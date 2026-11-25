# Ρομποτική Όραση, Διάλεξη 2: Σχηματισμός εικόνας, χρώμα και NumPy.
# Μικρά παραδείγματα για τις έννοιες της Python και τις συναρτήσεις της διάλεξης.
# Κάθε κελί (# %%) τρέχει μόνο του. Δεν χρειάζεται αρχείο εικόνας.

# %% 0. Εισαγωγές και μια συνθετική έγχρωμη εικόνα
import cv2
import numpy as np
import matplotlib.pyplot as plt


def make_scene(h=120, w=160):
    """Έγχρωμη εικόνα (BGR): γκριζοπράσινο φόντο, ένα μπλε ορθογώνιο και ένας κόκκινος κύκλος."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :] = (180, 200, 190)                               # φόντο, σε σειρά B, G, R
    cv2.rectangle(img, (20, 20), (70, 70), (200, 80, 30), -1)  # -1: γεμάτο σχήμα
    cv2.circle(img, (110, 60), 28, (30, 30, 210), -1)
    return img


img = make_scene()
h, w = img.shape[:2]
print("σχήμα:", img.shape, "| τύπος:", img.dtype)

# %% 1. Βασικά της Python: μεταβλητές, tuple, λίστα, λεξικό, if, for, while
pixel = (10, 200, 30)                       # tuple: σταθερό, δεν αλλάζει (B, G, R)
resolutions = [(640, 480), (1280, 720), (1920, 1080)]   # λίστα από tuples
names = {"B": 0, "G": 1, "R": 2}            # λεξικό: όνομα -> θέση

b, g, r = pixel                             # «ξεπακετάρισμα» tuple σε τρεις μεταβλητές
if g > b and g > r:                         # συνθήκη
    print("το pixel είναι κυρίως πράσινο, G =", g)

for width, height in resolutions:           # for πάνω σε λίστα
    print(f"{width}x{height}: {width * height / 1e6:.2f} Mpixel")   # f-string: μορφοποίηση

size = 1920
steps = 0
while size > 100:                           # while: επανάληψη όσο ισχύει η συνθήκη
    size //= 2                              # ακέραια διαίρεση
    steps += 1
print("υποδιπλασιάσαμε το πλάτος", steps, "φορές μέχρι", size, "pixels")
print("το κανάλι G έχει θέση", names["G"], "και τιμή", pixel[names["G"]])

# %% 2. Συναρτήσεις: ορίσματα, προεπιλεγμένες τιμές, πολλαπλές τιμές, lambda, list comprehension
def brightness(b, g, r, weights=(0.114, 0.587, 0.299)):
    """Φωτεινότητα ενός pixel. Το weights έχει προεπιλεγμένη τιμή (προαιρετικό όρισμα)."""
    return weights[0] * b + weights[1] * g + weights[2] * r


def min_max(values):
    """Επιστρέφει δύο τιμές μαζί (tuple)."""
    return min(values), max(values)


print("φωτεινότητα:", round(brightness(10, 200, 30), 1))
print("ίσα βάρη   :", round(brightness(10, 200, 30, weights=(1/3, 1/3, 1/3)), 1))   # όρισμα με όνομα

lo, hi = min_max(pixel)
print("min, max:", lo, hi)

double = lambda v: 2 * v                    # lambda: μικρή ανώνυμη συνάρτηση
print([double(v) for v in pixel])           # list comprehension: νέα λίστα από την παλιά
print([v for v in pixel if v > 20])         # με φίλτρο: κρατά μόνο τις τιμές > 20

# %% 3. Πίνακες NumPy: δημιουργία, σχήμα, τύπος, πράξεις
zeros = np.zeros((2, 3), np.uint8)          # όλα 0
ones = np.ones((2, 3), np.uint8)            # όλα 1
filled = np.full((2, 3), 7, np.uint8)       # όλα 7
steps_ = np.arange(0, 10, 2)                # [0 2 4 6 8]
ramp = np.linspace(0, 255, 5)               # 5 ίσα διαστήματα από 0 έως 255
print(zeros.shape, ones.dtype, filled[0, 0], steps_, ramp)

a = np.arange(12).reshape(3, 4)             # από 1D σε 2D
print(a)
print("ndim:", a.ndim, "| size:", a.size, "| άθροισμα:", a.sum())
print("ανά στήλη (axis=0):", a.sum(axis=0), "| ανά γραμμή (axis=1):", a.sum(axis=1))
print("πολλαπλασιασμός με αριθμό:", (a * 2)[0])            # πράξη σε ολόκληρο τον πίνακα
print("broadcasting (στήλη + γραμμή):", (a + np.array([100, 200, 300, 400]))[0])

# %% 4. Η εικόνα ως πίνακας: pixel, slicing, κανάλια, μάσκες
print("pixel (γραμμή 60, στήλη 110):", img[60, 110], "(BGR)")   # ΠΡΩΤΑ y, ΜΕΤΑ x
crop = img[20:70, 20:70]                    # αποκοπή [γραμμές, στήλες]
blue_channel = img[:, :, 0]                 # όλα τα pixels, κανάλι 0 (B)
flipped = img[:, ::-1]                      # κάθε στήλη με αντίστροφη σειρά = καθρέφτης
print("crop:", crop.shape, "| κανάλι:", blue_channel.shape, "| καθρέφτης:", flipped.shape)

mask = (img[:, :, 2] > 150) & (img[:, :, 0] < 100)   # True όπου R > 150 ΚΑΙ B < 100 (κόκκινο)
print("pixels του κόκκινου αντικειμένου:", mask.sum(), "από", mask.size)
red_only = img.copy()
red_only[~mask] = 0                         # ~mask: το αντίθετο. Μαυρίζουμε όλα τα υπόλοιπα
print("μη μηδενικά pixels:", np.count_nonzero(red_only.any(axis=2)))

# %% 5. Από το φως στο pixel: ένταση, ανακλαστικότητα, αισθητήρας 8 bit
light = np.array([0.1, 0.4, 1.0, 2.0])      # σχετική ένταση φωτός (2.0 = υπερέκθεση)
reflectance = 0.6                           # πόσο ανακλά η επιφάνεια
signal = light * reflectance                # πράξη σε ολόκληρο τον πίνακα
pixels = np.clip(np.round(signal * 255), 0, 255).astype(np.uint8)   # κόψιμο στο 0–255
print("τιμές pixel:", pixels, "(το 2.0 «καίγεται» στο 255)")

a = np.array([200], np.uint8)
print("λάθος: 200 + 80 =", (a + 80)[0], "| σωστά με clip:",
      np.clip(a.astype(np.int16) + 80, 0, 255).astype(np.uint8)[0])

# %% 6. Δειγματοληψία: λιγότερα pixels
small = cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_AREA)   # μέσος όρος ανά περιοχή
sub = img[::4, ::4]                                                       # κάθε 4ο pixel
blocky = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)       # μεγέθυνση «με τετράγωνα»
print("αρχική:", img.shape, "| INTER_AREA:", small.shape, "| slicing:", sub.shape, "| μεγέθυνση:", blocky.shape)

# %% 7. Κβαντισμός: λιγότερα επίπεδα φωτεινότητας
ramp = np.tile(np.arange(256, dtype=np.uint8), (10, 1))      # βαθμίδα 0..255


def quantize(image, levels):
    step = 256 // levels
    return (image // step) * step           # // : ακέραια διαίρεση


for bits in (8, 4, 2, 1):                   # bits -> πλήθος επιπέδων = 2 ** bits
    q = quantize(ramp, 2 ** bits)
    print(f"{bits} bit: {2 ** bits} επίπεδα, διαφορετικές τιμές στην εικόνα: {len(np.unique(q))}")

# %% 8. Χρωματικοί χώροι: BGR, RGB, γκρι, HSV
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
b, g, r = cv2.split(img)                                    # χωρισμός καναλιών
print("merge(split) = αρχική:", np.array_equal(cv2.merge([b, g, r]), img))
print("RGB = BGR αντεστραμμένο:", np.array_equal(rgb, img[:, :, ::-1]))

gray_manual = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)     # ο τύπος της φωτεινότητας
print("μέγιστη διαφορά από το cv2:", np.abs(gray.astype(int) - gray_manual.astype(int)).max())

colors = {"κόκκινο": (0, 0, 255), "πράσινο": (0, 255, 0), "μπλε": (255, 0, 0)}   # BGR
for name, bgr in colors.items():                            # for πάνω σε λεξικό
    h_, s_, v_ = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
    print(f"{name:8s} BGR{bgr} -> H={h_}, S={s_}, V={v_}   (στην OpenCV το H είναι 0–179)")

# %% 9. Γεωμετρικοί μετασχηματισμοί: κλίμακα, περιστροφή, μετατόπιση, καθρέφτης
scaled = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
M = cv2.getRotationMatrix2D((w // 2, h // 2), 30, 1.0)     # (κέντρο, γωνία σε μοίρες, κλίμακα)
rotated = cv2.warpAffine(img, M, (w, h))
T = np.float32([[1, 0, 30], [0, 1, 10]])                   # 30 pixels δεξιά, 10 κάτω
shifted = cv2.warpAffine(img, T, (w, h))
print("κλίμακα 50%:", scaled.shape)

p = np.array([w // 2 + 20, h // 2, 1])                     # σημείο 20 pixels δεξιά του κέντρου (x, y, 1)
print("νέα θέση του σημείου μετά την περιστροφή:", np.round(M @ p, 1), "(περίπου [97.3 50.0])")
print("μετατόπιση του ίδιου σημείου:", T @ p)             # M @ p: πολλαπλασιασμός πίνακα επί διάνυσμα

print("cv2.flip = slicing:", np.array_equal(cv2.flip(img, 1), img[:, ::-1]))
print("περιστροφή 90°:", cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE).shape, "(ύψος και πλάτος αντιστρέφονται)")

# %% 10. Εμφάνιση με Matplotlib (προαιρετικό)
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for ax, (im, title) in zip(axes, [(rgb, "RGB"), (gray, "Γκρι"), (rotated[:, :, ::-1], "Περιστροφή 30°"),
                                  (blocky[:, :, ::-1], "Δειγματοληψία 1/4")]):
    ax.imshow(im, cmap="gray" if im.ndim == 2 else None)   # το Matplotlib θέλει RGB, όχι BGR
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()
