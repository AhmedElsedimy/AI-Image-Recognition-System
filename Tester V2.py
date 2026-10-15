import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import piexif
import os

# Settings
num_classes = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = 'best_model.pth'
test_image_dir = 'C:/Users/HP/Desktop/TT'
class_names = ['Japan', 'Türkiye', 'United Arab Emirates', 'United Kingdom']

# Load model
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
#model.load_state_dict(torch.load(model_path))
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model = model.to(device)
model.eval()

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def display_image_with_title(image_path, title):
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show(block=False)
    plt.pause(2)  # Show for 2 seconds
    plt.close()

# def write_country_to_metadata(image_path, country):
#     try:
#         img = Image.open(image_path)
#         exif_dict = piexif.load(img.info.get('exif', b''))
#         if "0th" not in exif_dict:
#             exif_dict["0th"] = {}
#
#         exif_dict["0th"][piexif.ImageIFD.ImageDescription] = country.encode('utf-8')
#         exif_bytes = piexif.dump(exif_dict)
#         img.save(image_path, "jpeg", exif=exif_bytes)
#     except Exception as e:
#         print(f"Failed to write metadata for {image_path}: {e}")
def write_country_to_metadata(image_path, country):
    try:
        img = Image.open(image_path)

        # Load existing EXIF data if available, else start from empty dict
        try:
            exif_dict = piexif.load(img.info["exif"])
        except (KeyError, piexif.InvalidImageDataError):
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

        # Set country in ImageDescription (ID = 270)
        exif_dict["0th"][piexif.ImageIFD.ImageDescription] = country.encode('utf-8')
        exif_bytes = piexif.dump(exif_dict)

        # Re-save image with new EXIF data (must be JPEG!)
        img.save(image_path, "jpeg", exif=exif_bytes)
    except Exception as e:
        print(f"Failed to write metadata for {image_path}: {e}")


def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    predicted_country = class_names[pred.item()]
    confidence = conf.item() * 100
    print(f"{os.path.basename(image_path)} ➔ Predicted: {predicted_country} ({confidence:.2f}%)")

    display_image_with_title(image_path, f"{predicted_country} ({confidence:.1f}%)")
    write_country_to_metadata(image_path, predicted_country)

# Predict all images in folder
for img_name in os.listdir(test_image_dir):
    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(test_image_dir, img_name)
        predict_image(img_path)
