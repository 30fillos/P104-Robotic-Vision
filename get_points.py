
import cv2

 
# Φόρτωση φωτογραφίας
img = cv2.imread("data/1st_Ass/scene1_normal.jpg")

# Ίδια προετοιμασία με το starter
h, w = img.shape[:2]
new_h = round(h * 640 / w)

img = cv2.resize(img, (640, new_h))

# Συνάρτηση καταγραφής κλικ
def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"X = {x}, Y = {y}")

# Δημιουργία παραθύρου
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Image", mouse_click)

print("Κάνε κλικ στα κέντρα των αντικειμένων.")
print("Πάτησε ESC για έξοδο.")

while True:
    cv2.imshow("Image", img)

    key = cv2.waitKey(1)

    if key == 27:
        break

cv2.destroyAllWindows()
