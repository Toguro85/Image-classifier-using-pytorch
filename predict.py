import argparse
from image_classifier_project import *
from PIL import Image
import matplotlib.pyplot as plt
import json

def parse_input_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('input',action='store',type='str')
    parser.add_argument('checkpoint',action='store',type='str',default='checkpoint.pth')
    parser.add_argument('--top_k',type=int,default=5)
    parser.add_argument('--category_names',type='str',default='')
    parser.add_argument('--gpu',action='store_true',default=False)
    parser.add_argument('--input_category',type='str',default='')
    return parser.parse_args()
def main():
    in_arg=parse_input_args()
    model,accuracy,learnrate=checkpoint_load(in_arg.checkpoint)
    if in_arg.category_names!='':
        with open(in_arg.category_names,'r') as f:
            cat_to_name=json.load(f)
        model.class_to_idx=cat_to_name
    if in_arg.gpu==True:
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device="cpu"
    print("Predict Image")
    
    probability_t,class_t=predict(in_arg.input,model,device,in_arg.top_k)
    probability=probability_t.tolist()[0]
    if in_arg.category_names!='':
        classes=[model.class_to_idx[str(sorted(model.class_to_idx)[i])] for i in (class_t).tolist()[0]]
    else:
        classes=class_t.tolist()[0]
    
    print("\nInput image : {}".format(in_arg.input))
    print("\nInput variety : {}".format(in_arg.input_category))
    print("\nPredicted variety : {}".format(classes[0]))
    image=process_image(Image.open(in_arg.input))
    
if __name__ =='__main__':
    main()