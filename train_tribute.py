import os

from guandanAI.tribute import parser
from guandanAI.tribute import train_tribute

if __name__ == '__main__':
    flags = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu_devices
    train_tribute(flags)