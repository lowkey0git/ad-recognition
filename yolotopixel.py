def yolo_to_pixel(box, img_w, img_h):
    _, x, y, w, h = box
    x1 = (x - w / 2) * img_w
    y1 = (y - h / 2) * img_h
    x2 = (x + w / 2) * img_w
    y2 = (y + h / 2) * img_h
    return int(x1), int(y1), int(x2), int(y2)
