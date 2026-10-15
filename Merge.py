import os
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import piexif
import tkinter as tk
from tkinter import filedialog
from io import BytesIO

# --- إعدادات النماذج ---
num_classes = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path_fasterrcnn = 'C:\\Users\\HP\\Downloads\\26April7.pth'

num_country_classes = 4
model_path_resnet50 = 'C:\\Users\\HP\\Downloads\\best_model.pth'
class_names = ['Japan', 'Türkiye', 'United Arab Emirates', 'United Kingdom']

# --- تحميل النماذج ---

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def get_resnet_model():
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_country_classes)
    model.load_state_dict(torch.load(model_path_resnet50, map_location=device))
    model.eval()
    return model

model_fasterrcnn = get_model(num_classes)
model_fasterrcnn.load_state_dict(torch.load(model_path_fasterrcnn, map_location=device))
model_fasterrcnn.to(device)
model_fasterrcnn.eval()

model_resnet50 = get_resnet_model()
model_resnet50.to(device)

# --- التحويلات ---
transform_resnet50 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --- الكلاسات ---
COCO_CLASSES = {0: "Objects", 1: "Build", 2: "Person", 3: "Sign", 4: "Store", 5: "Street", 6: "Vegetation", 7: "Vehicle"}

def get_class_name(class_id):
    return COCO_CLASSES.get(class_id, "Unknown")

def prepare_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0)
    return image_tensor.to(device)

def write_country_to_metadata(image_path, country):
    try:
        img = Image.open(image_path)
        try:
            exif_dict = piexif.load(img.info.get("exif", b""))
        except (KeyError, piexif.InvalidImageDataError):
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
        exif_dict["0th"][piexif.ImageIFD.ImageDescription] = country.encode('utf-8')
        exif_bytes = piexif.dump(exif_dict)
        img.save(image_path, "jpeg", exif=exif_bytes)
    except Exception as e:
        print(f"Failed to write metadata for {image_path}: {e}")

def get_country_name(image):
    image_resnet = transform_resnet50(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model_resnet50(image_resnet)
        probs = torch.softmax(outputs, dim=1)
        _, pred = torch.max(probs, 1)
    predicted_country = class_names[pred.item()]
    return predicted_country

def draw_boxes_on_image(image, prediction, country_name):
    boxes = prediction[0]['boxes'].detach().cpu().numpy()
    labels = prediction[0]['labels'].detach().cpu().numpy()
    scores = prediction[0]['scores'].detach().cpu().numpy()

    threshold = 0.5
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    ax = plt.gca()
    ax.text(10, 10, f"Country: {country_name}", color='blue', fontsize=14, bbox=dict(facecolor='white', alpha=0.7))

    for box, label, score in zip(boxes, labels, scores):
        if score > threshold:
            x_min, y_min, x_max, y_max = box
            class_name = get_class_name(label)
            ax.add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                       linewidth=2, edgecolor='r', facecolor='none'))
            ax.text(x_min, y_min, f"{class_name} ({score:.2f})", color='red', fontsize=8,
                    bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    return Image.open(buf)

# --- شاشة الترحيب ---
class SplashScreen:
    def __init__(self, start_callback):
        self.root = tk.Tk()
        self.root.title("مرحبا بك")
        self.root.geometry("600x400")
        self.root.configure(bg='white')
        self.start_callback = start_callback

        # تحميل الشعار (غيره حسب مسار صورتك!)
        self.logo_image = Image.open('C:\\Users\\HP\\Downloads\\png-clipart-pharos-university-in-alexandria-alexandria-higher-institute-of-engineering-and-technology-dublin-institute-of-technology-nile-university-science-and-technology-emblem-label.png')  # تأكد أن لديك ملف شعار اسمه logo.png بجانب السكربت
        self.logo_image = self.logo_image.resize((150, 150))
        self.tk_logo = ImageTk.PhotoImage(self.logo_image)

        self.logo_label = tk.Label(self.root, image=self.tk_logo, bg='white')
        self.logo_label.pack(pady=20)

        self.name_label = tk.Label(self.root, text="DR.Wissam Abdulsalam", font=("Arial", 20, 'bold'), fg='blue', bg='white')
        self.name_label.pack()

        self.project_label = tk.Label(self.root, text="Country Street View", font=("Arial", 16), fg='green', bg='white')
        self.project_label.pack()
        self.project_label = tk.Label(self.root, text="(Faster R-CNN with Resnet50)", font=("Arial", 16), fg='green', bg='white')
        self.project_label.pack()

        self.start_button = tk.Button(self.root, text="Start", font=("Arial", 16), command=self.start_program)
        self.start_button.pack(pady=30)

        # تأثير ظهور تدريجي
        self.alpha = 0.0
        self.fade_in()

        self.root.mainloop()

    def fade_in(self):
        if self.alpha < 1.0:
            self.alpha += 0.02
            self.root.attributes('-alpha', self.alpha)
            self.root.after(30, self.fade_in)

    def start_program(self):
        self.root.destroy()
        self.start_callback()

# --- مستعرض الصور ---
class ImageViewer:
    def __init__(self):
        self.folder_path = filedialog.askdirectory(title="اختر مجلد الصور")
        if not self.folder_path:
            print("لم يتم اختيار مجلد، إنهاء البرنامج.")
            exit()

        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        self.image_files = [f for f in os.listdir(self.folder_path) if any(f.lower().endswith(ext) for ext in self.image_extensions)]
        self.index = 0

        self.root = tk.Tk()
        self.root.title("Image Viewer")
        self.label = tk.Label(self.root)
        self.label.pack()

        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack()

        self.prev_button = tk.Button(self.button_frame, text="<< Previous", command=self.show_previous_image)
        self.prev_button.pack(side=tk.LEFT, padx=5)

        self.next_button = tk.Button(self.button_frame, text="Next >>", command=self.show_next_image)
        self.next_button.pack(side=tk.LEFT, padx=5)

        self.info_label = tk.Label(self.root, text="", font=("Arial", 12))
        self.info_label.pack(pady=10)

        self.show_image()

        self.root.mainloop()

    def process_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        country_name = get_country_name(image)
        image_tensor = prepare_image(image_path)
        with torch.no_grad():
            prediction = model_fasterrcnn(image_tensor)
        final_image = draw_boxes_on_image(image, prediction, country_name)
        write_country_to_metadata(image_path, country_name)
        return final_image

    def show_image(self):
        if 0 <= self.index < len(self.image_files):
            image_path = os.path.join(self.folder_path, self.image_files[self.index])
            print(f"Processing: {image_path}")
            processed_image = self.process_image(image_path)

            self.tk_image = ImageTk.PhotoImage(processed_image)
            self.label.config(image=self.tk_image)
            self.label.image = self.tk_image

            width, height = processed_image.size
            self.root.geometry(f"{width}x{height+100}")

            self.info_label.config(text=f"Image {self.index + 1} of {len(self.image_files)}")

    def show_next_image(self):
        if self.index < len(self.image_files) - 1:
            self.index += 1
            self.show_image()

    def show_previous_image(self):
        if self.index > 0:
            self.index -= 1
            self.show_image()

# --- بدء تشغيل البرنامج ---

def start_app():
    ImageViewer()

if __name__ == "__main__":
    SplashScreen(start_callback=start_app)
