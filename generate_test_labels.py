'''
This code is used to evaluate the classification accuracy of the trained model.
You should at least guarantee this code can run without any error on validation set.
And whether this code can run is the most important factor for grading.
We provide the remaining code, all you should do are, and you can't modify other code:
1. Replace the random classifier with your trained model.(line 69-72)
2. modify the get_label function to get the predicted label.(line 23-29)(just like Leetcode solutions, the args of the function can't be changed)

REQUIREMENTS:
- You should save your model to the path 'models/conditional_pixelcnn.pth'
- You should Print the accuracy of the model on validation set, when we evaluate your code, we will use test set to evaluate the accuracy
'''
from torchvision import datasets, transforms
from utils import *
from model import * 
from dataset import *
from classification_evaluation import *
from tqdm import tqdm
from pprint import pprint
import argparse
import csv
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=32, help='Batch size for inference')
    parser.add_argument('-m', '--mode', type=str,
                        default='test', help='Mode for the dataset')
    
    args = parser.parse_args()
    pprint(args.__dict__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers':0, 'pin_memory':True, 'drop_last':False}

    ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling])
    test_dataset = CPEN455Dataset_test(root_dir=args.data_dir, mode = args.mode, transform=ds_transforms)
    num_samples = len(test_dataset)

    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=num_samples, shuffle=False, **kwargs)

    model = ConditionalPixelCNN()
    
    model = model.to(device)
    MODEL_NAME = 'models/conditional_pixelcnn.pth'
    model_path = os.path.join(os.path.dirname(__file__), MODEL_NAME)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print('model parameters loaded')
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model.eval()

    output_file='predictions/test_predictions.csv'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    for batch_idx, item in enumerate(tqdm(dataloader)):
        model_input, _, file_paths = item # no class labels for test set, predict most likely class for each input sample
        # pdb.set_trace()
        model_input = model_input.to(device)
        
        predicted_classes = get_label(model, model_input, device)
        predicted_classes_int = predicted_classes.cpu().tolist()

        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)            
            for filename, pred_class in zip(file_paths, predicted_classes_int):
                writer.writerow([f"{filename}", pred_class])


    