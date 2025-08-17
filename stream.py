import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# ==========================
# Config
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

num_classes = 3
checkpoint_path = r"C:\Users\abdul\OneDrive\Documents\images_class\image_classification_model.pth"

# Correct mapping from training
class_names = ["davido", "rihanna", "ronaldo"]

# ==========================
# Model Definition (same as training)
# ==========================
model = models.resnet50(pretrained=True)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, num_classes)
)

# Load checkpoint
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model = model.to(device)
model.eval()
print("✅ Model loaded successfully!")

# ==========================
# Preprocessing
# ==========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_face(face_img):
    """Classify cropped face image with confidence score"""
    image = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)  # get probabilities
        confidence, predicted = torch.max(probs, 1)

    label = class_names[predicted.item()]
    conf = confidence.item() * 100  # percentage
    return f"{label} ({conf:.1f}%)"

# ==========================
# OpenCV Face Detection + Real-time Classification
# ==========================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)  # webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]  # crop
        label = predict_face(face_img)

        # Draw bounding box + label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

    cv2.imshow("Real-Time Face Classification", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()