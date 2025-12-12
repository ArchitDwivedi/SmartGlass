def build_caption(detections):
    if not detections:
        return "No objects detected. Path is clear."

    caption_parts = []
    for obj in detections:
        name = obj["name"]
        dist = obj["distance"]

        # Distance in natural spoken words
        if dist < 1.0:
            label = "very close"
        elif dist < 2.5:
            label = "near"
        else:
            label = "far"

        caption_parts.append(f"{name} {label} at {dist:.1f} meters")

    return ", ".join(caption_parts)
