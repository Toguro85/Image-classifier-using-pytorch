import argparse
from image_classifier_project import *
def main():
    in_arg=parse_input_args()
    if in_arg.hidden_units==[]:
        in_arg.hidden_units=[4096,4096];
    save_path=in_arg.save_dir+in_arg.checkpoint
    hyperparameters={'learnrate':in_arg.learning_rate,
                     'hidden_layers':in_arg.hidden_units,
                     'epochs':in_arg.epochs,
                     'architecture':in_arg.arc,
                     'dropout_probability':in_arg.drop_p}
    #user can choose from (alexnet,vgg16,vgg13,vgg11)
    #further models can be added but would require change in jupyter notebook as well as image_classifier_project.py
    #so submitting with the above 4 as otherwise i will have to train the model again and go through
    #the whole process again,these were included in the previous submission as well
    model=get_model(hyperparameters['architecture'])
    model,train_loader,test_loader=model_config(in_arg.data_dir,model,hyperparameters['hidden_layers'],
                                                ['dropout_probability'])
    if in_arg.gpu==True:
        device=torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    else:
        device='cpu'
    
    print("Epochs:   {}".format(hyperparameters['epochs']))
    print("\n")
    model_accuracy=train(model,train_dataloader,test_dataloader,device,
                         hyperparameters['learnrate'],epochs=hyperparameters['epochs'],
                         print_every=40,
                         debug=0)
    checkpoint_save(save_path,model,hyperparameters['architecture'],model.classifier.hidden_layers[0].in_features,
                    model.classifier.output.out_features,hyperparameters,accuracy)

def parse_input_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('data_dir',action='store',type=str,default='./flowers/',
                        help='path to data directory')
    parser.add_argument('--save_dir',type=str,default='./model_check_points/',
                        help='path to model checkpoints')
    parser.add_argument('--checkpoint',type=str,default='checkpoint.pth',
                        help='name of checkpoint file')
    parser.add_argument('--arch',type=str,default='vgg16',
                        help='model name')
    parser.add_argument('--learning_rate',type=float,default=0.001,
                        help='learning rate')
    parser.add_argument('--hidden_units',action='append',type=int,default=[],
                        help='hidden layer units')
    parser.add_argument('--epochs',type=int,default=5,
                        help='epochs')
    parser.add_argument('--drop_p',type=float,default=0.2,
                        help='dropout probability')
    parser.add_argument('--gpu',action='store_true',default='False',
                        help='Gpu available')
    return parser.parse_args()

if __name__=='__main__':
    main()
