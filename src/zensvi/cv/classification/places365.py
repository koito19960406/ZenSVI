from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms as trn
from pathlib import Path

from .base import BaseClassifier

class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.img_paths = list(self.img_dir.glob('*.[pj][np][g]'))  # Supports .jpg, .jpeg, .png
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = read_image(str(img_path))
        if self.transform:
            image = self.transform(image)
        return image, str(img_path)

    def collate_fn(self, batch):
        images, paths = zip(*batch)
        return list(images), list(paths)

def returnTF():
    # load the image transformer
    tf = trn.Compose([
        trn.Resize((224, 224)),
        trn.ConvertImageDtype(torch.float),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return tf

class ClassifierPlaces365(BaseClassifier):
    def __init__(self, device = None):
        super().__init__(device)
        self.device = self._get_device(device)
        self.classes, self.labels_IO, self.labels_attribute, self.W_attribute = load_labels()
        self.features_blobs = []
        self.model = load_model()
        self.model.eval()
        self.model.to(self.device)
        self.hook = self.model._modules.get('layer4').register_forward_hook(hook_feature)
    
    def _load_labels(self):
        # prepare all the labels
        # scene category relevant
        file_name_category = 'categories_places365.txt'
        classes = list()
        with open(file_name_category) as class_file:
            for line in class_file:
                classes.append(line.strip().split(' ')[0][3:])
        classes = tuple(classes)

        # indoor and outdoor relevant
        file_name_IO = 'IO_places365.txt'
        with open(file_name_IO) as f:
            lines = f.readlines()
            labels_IO = []
            for line in lines:
                items = line.rstrip().split()
                labels_IO.append(int(items[-1]) - 1)  # 0 is indoor, 1 is outdoor
        labels_IO = np.array(labels_IO)

        # scene attribute relevant
        file_name_attribute = 'labels_sunattribute.txt'
        with open(file_name_attribute) as f:
            lines = f.readlines()
            labels_attribute = [item.rstrip() for item in lines]
        file_name_W = 'W_sceneattribute_wideresnet18.npy'
        W_attribute = np.load(file_name_W)

        return classes, labels_IO, labels_attribute, W_attribute
        
def classify_places365(self, 
                    dir_input: Union[str, Path], 
                    dir_image_output: Union[str, Path, None] = None, 
                    dir_summary_output: Union[str, Path, None] = None, 
                    batch_size=1, 
                    save_image_options = ["cam_image", "blend_image"], 
                    pixel_ratio_save_format = ["json", "csv"],
                    csv_format = "long", # "long" or "wide"
                ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Prepare DataLoader
    dataset = ImageDataset(img_dir, transform=returnTF())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Process images in batches
    for inputs, paths in tqdm(dataloader, desc="Processing images"):
        # inputs: batch of images, paths: batch of image paths
        # Implement batch processing logic here
        pass
