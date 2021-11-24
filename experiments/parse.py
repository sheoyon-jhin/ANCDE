import argparse

# argparser -> model name, hidden_channels, etc
# main(intensity = True, model_name='ncde',hidden_channels=49, hidden_hidden_channels=49, num_hidden_layers=4)

def parse_args():
    parser = argparse.ArgumentParser(description='Attentive CDE')
    parser.add_argument('--seed', type=int, default=2021,help='Seed - Test your luck!')   
    parser.add_argument('--intensity', type=bool, default=True,help='Intensity')
    parser.add_argument('--model', type=str, default='ncde',help='Model Name')
    parser.add_argument('--h_channels', type=int, default=49,help='Hidden Channels')     
    parser.add_argument('--hh_channels', type=int, default=49,help='Hidden Hidden Channels')          
    parser.add_argument('--layers', type=int, default=4,help='Num of Hidden Layers')   
    parser.add_argument('--lr', type=float, default=0.0001,help='Learning Rate')  
    parser.add_argument('--epoch',type=int,default = 200,help ='Epoch') 
    parser.add_argument('--init_channels', type = int, default = 4,help = "init channels for attention")
    parser.add_argument('--slope_check',type=bool,default =True,help="Slope")
    parser.add_argument('--soft',type=bool,default =True,help="Soft Attention")
    parser.add_argument('--timewise',type=bool,default =True,help="Timewise Attention")
    parser.add_argument('--attention_channel', type=int, default=20,help='First Attention Hidden vector size')
    parser.add_argument('--attention_attention_channel',type=int,default=10,help='Second Attention Hidden vector size')
    parser.add_argument('--step_mode', type=str, default='valloss',help='Model Name')
    parser.add_argument('--dataset_name', type=str, default='CharacterTrajectories',help='Dataset Name')
    parser.add_argument('--missing_rate', type=float, default=0.3,help='Missing Rate')
    parser.add_argument('--c1', type=float, default=0,help='mse loss coefficient')
    parser.add_argument('--c2', type=float, default=0,help='Hutchinson coefficient')
    parser.add_argument('--rtol', type=float, default=1e-11,help='ODEINT  Rtol ')
    parser.add_argument('--atol', type=float, default=1e-11,help='Hutchinson coefficient')
    parser.add_argument('--sequence', type=int, default=24,help='Time sequence')
    
    # parser.add_argyment('--')
    return parser.parse_args()