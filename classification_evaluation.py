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
from tqdm import tqdm
from pprint import pprint
import argparse
import csv
NUM_CLASSES = len(my_bidict)

#TODO: Begin of your code
def get_label(model, model_input, device):
    # Write your code here, replace the random classifier with your trained model
    # and return the predicted label, which is a tensor of shape (batch_size,)
    batch_size = model_input.size(0)
    model.eval()
    
    # For each sample, compute Neg log likelihood for each of the four class condition
    with torch.no_grad():
        # will be a Bx4 matrix
        all_nlls = torch.zeros(batch_size, NUM_CLASSES, device=device)
        
        for class_idx in range(NUM_CLASSES):
            # create a tensor with the same class label for the entire batch
            class_labels = torch.full((batch_size,), class_idx, dtype=torch.long, device=device)
            
            # run particular class condition through model
            output = model(model_input, class_labels)
            
            # get the NLL for each sample
            for i in range(batch_size):
                single_input = model_input[i:i+1]
                single_output = output[i:i+1]
                nll = discretized_mix_logistic_loss(single_input, single_output)
                all_nlls[i, class_idx] = nll
    
    # get most probable out of the all classes (since NLL is negative, take min to get the max likelihood)
    _, predicted_labels = torch.min(all_nlls, dim=1)
    
    return predicted_labels
# End of your code

def classifier(model, data_loader, device):
    model.eval()
    acc_tracker = ratio_tracker()
    for batch_idx, item in enumerate(tqdm(data_loader)):
        model_input, categories = item
        model_input = model_input.to(device)
        original_label = [my_bidict[item] for item in categories]
        original_label = torch.tensor(original_label, dtype=torch.int64).to(device)
        answer = get_label(model, model_input, device)
        correct_num = torch.sum(answer == original_label)
        acc_tracker.update(correct_num.item(), model_input.shape[0])
    
    return acc_tracker.get_ratio()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=32, help='Batch size for inference')
    parser.add_argument('-m', '--mode', type=str,
                        default='validation', help='Mode for the dataset')
    
    args = parser.parse_args()
    pprint(args.__dict__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers':0, 'pin_memory':True, 'drop_last':False}

    ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling])
    dataloader = torch.utils.data.DataLoader(CPEN455Dataset(root_dir=args.data_dir, 
                                                            mode = args.mode, 
                                                            transform=ds_transforms), 
                                             batch_size=args.batch_size, 
                                             shuffle=True, 
                                             **kwargs)

    #TODO:Begin of your code
    model = ConditionalPixelCNN()
    #End of your code
    
    model = model.to(device)
    #Attention: the path of the model is fixed to './models/conditional_pixelcnn.pth'
    #You should save your model to this path
    
    MODEL_NAME = 'models/conditional_pixelcnn.pth'
    # MODEL_NAME = 'models/pcnn_cpen455_film_output_learned_gamma_from_scratch_0.pth'
    model_path = os.path.join(os.path.dirname(__file__), MODEL_NAME)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print('model parameters loaded')
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model.eval()
    
    acc = classifier(model = model, data_loader = dataloader, device = device)
    print(f"Accuracy: {acc}")
        
        