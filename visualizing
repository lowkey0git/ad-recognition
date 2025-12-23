model = MultiObjectDetector(num_classes=2)
model.load_state_dict(
    torch.load("/content/object_detector.pth")
)
model.eval()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

# 🔹 LOAD IMAGE TO EVALUATE
img_path = "/content/testing"
img = Image.open(img_path).convert("RGB")
w, h = img.size

x = transform(img).unsqueeze(0)

with torch.no_grad():
    preds = model(x)[0]

img_cv = cv2.cvtColor(
    cv2.imread(img_path),
    cv2.COLOR_BGR2RGB
)

for obj in preds:
    class_id = torch.argmax(obj[:2]).item()
    box = obj.numpy()

    x1, y1, x2, y2 = yolo_to_pixel(box, w, h)

    cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        img_cv,
        f"Class {class_id}",
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )

cv2.imshow("AI Recognition", img_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()
