import os 
import argparse

from guandanAI.evaluation.simulation import evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    'GuanDan Evaluation')
    parser.add_argument('--player_1', type=str,
            default='models/model.ckpt')
    parser.add_argument('--player_2', type=str,
            default='models/model.ckpt')
    parser.add_argument('--player_3', type=str,
            default='models/model.ckpt')
    parser.add_argument('--player_4', type=str,
            default='models/model.ckpt')
    
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--gpu_device', type=str, default='')
    args = parser.parse_args()

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
#     _devices = args.gpu_device.split(",")
#     devices = ["cuda:"+device.strip() for device in _devices if device!="cpu"]
#     if "cpu" in _devices:
#         devices.append("cpu")
    evaluate([args.player_1, args.player_2, args.player_3, args.player_4,], args.num_workers, args.num_epochs)
