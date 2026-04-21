# 🔹 LOAD DATA for training
dataset = ObjectDataset(
    img_dir="/content/Banner",
    label_dir="/content/obj_Train_data"
)

loader = DataLoader(dataset, batch_size=2, shuffle=True)

# 🔹 MODEL
num_classes = 2  # change to your real class count
model = MultiObjectDetector(num_classes=num_classes)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
class_loss = nn.CrossEntropyLoss()
bbox_loss = nn.MSELoss()

# 🔹 TRAIN LOOP
for epoch in range(10):
    total_loss = 0

    for images, targets in loader:
        preds = model(images)

        loss = 0
        for i in range(preds.shape[1]):
            class_pred = preds[:, i, :num_classes]
            bbox_pred = preds[:, i, num_classes:]

            class_true = targets[:, i, 0].long()
            bbox_true = targets[:, i, 1:]

            loss += class_loss(class_pred, class_true)
            loss += bbox_loss(bbox_pred, bbox_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

# 🔹 SAVE MODEL
torch.save(model.state_dict(), "/content/object_detector.pth")
