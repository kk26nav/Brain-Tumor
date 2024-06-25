from ultralytics import YOLO
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
if __name__ == '__main__':
	# Load your trained model
	model = YOLO("C:/Users/snk20/Downloads/Brain Tumor/runs/classify/train11/weights/best.pt")

	# Define the dataset path
	dataset_path = "C:/Users/snk20/Downloads/Brain Tumor/cleanednew/test"

	# Define transformations
	transform = transforms.Compose([
		transforms.Resize((256, 256)),
		transforms.ToTensor()
	])

	# Load the dataset
	dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
	data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
	y_true = []
	y_pred = []
	
	# Iterate through the dataset
	for images, labels in data_loader:
		results = model(images)
		y_true.extend(labels.tolist())	
		for result in results:
			predictions = result.probs.top1  # Get top-1 predictions
			y_pred.extend(predictions.tolist() if hasattr(predictions, 'tolist') else [predictions])
	print("yt:",y_true,"Length=",len(y_true))	
	print("yr:",y_pred,"Length=",len(y_pred))	
	# Ensure y_true and y_pred have the same length
	assert len(y_true) == len(y_pred), f"Mismatch in lengths: y_true={len(y_true)}, y_pred={len(y_pred)}"
	# Calculate metrics
	precision = precision_score(y_true, y_pred, average='weighted')
	recall = recall_score(y_true, y_pred, average='weighted')
	f1 = f1_score(y_true, y_pred, average='weighted')
	report=classification_report(y_true, y_pred)
	# Calculate Specificity and Sensitivity
	cm = confusion_matrix(y_true, y_pred)
	print(cm)
	'''tn, fp, fn, tp = cm.ravel()
	specificity = tn / (tn + fp)
	sensitivity = tp / (tp + fn)'''

	print(f"Precision: {precision}")
	print(f"Recall: {recall}")
	print(f"F1-Score: {f1}")
	print(f"{report}")
	#print(f"Specificity: {specificity}")
	#print(f"Sensitivity: {sensitivity}")