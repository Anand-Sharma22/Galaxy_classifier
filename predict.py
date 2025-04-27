import torch
from torchvision import transforms
from PIL import Image
from config import resize_x, resize_y
from model import GCNN

def predict_model(list_of_img_paths, device= torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu") ):
    model = GCNN().to(device)
    model.load_state_dict(torch.load('checkpoints/final_weights.pth', map_location=device))
    model.eval()
    images = []
    transform = transforms.Compose([
        transforms.Resize((resize_x, resize_y)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    for path in list_of_img_paths:
        img = Image.open(path).convert('RGB')
        img = transform(img)
        images.append(img)
    
    batch = torch.stack(images).to(device)
    
    with torch.no_grad():
        outputs = model(batch)
        preds = outputs.argmax(dim=1).cpu().numpy()
    
    labels = ["spiral" if p == 0 else "elliptical" for p in preds]
    return labels