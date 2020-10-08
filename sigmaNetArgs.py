import argparse
import os
import tensorflow as tf

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2int(v):
    if v=='M':
        return v
    try:
        v = int(v)
    except:
        raise argparse.ArgumentTypeError('int value expected.')
    return v
    
def str2float(v):
    if v=='M':
        return v
    try:
        v = float(v)
    except:
        raise argparse.ArgumentTypeError('float value expected.')
    return v
        
def args():
    # TODO modularise this into concentricGAN noise->clean->SR->seg->vels
    parser = argparse.ArgumentParser(description='')
    #training arguments
    parser.add_argument('--mixedPrecision', dest='mixedPrecision', type=str2bool, default=False, help='16bit computes')
    parser.add_argument('--gpuIDs', dest='gpuIDs', type=str, default='0,1,2,3', help='IDs for the GPUs. Empty for CPU. Nospaces')
    parser.add_argument('--nDims', dest='nDims', type=str2int, default=2, help='input dimensions')
    parser.add_argument('--dataset_dir', dest='dataset_dir', default='/media/user/SSD3/datasets/cycleSRANU_2D/', help='dataset path - include last slash')
    parser.add_argument('--augFlag', dest='augFlag', type=str2bool, default=True, help='aug flag')
    parser.add_argument('--epoch', dest='epoch', type=str2int, default=500, help='# of epoch')
    parser.add_argument('--scale', dest='scale', type=str2int, default=4, help='sr scale factor')
    parser.add_argument('--batch_size', dest='batch_size', type=str2int, default=64, help='# images in batch')
    parser.add_argument('--iterNum', dest='iterNum', type=str2int, default=21101, help='# iterations per epoch') 
    parser.add_argument('--itersPerEpoch', dest='itersPerEpoch', type=str2int, default=1000, help='# iterations per epoch') 
    parser.add_argument('--valNum', dest='valNum', type=str2int, default=25, help='# max val images') 
    parser.add_argument('--val_size', dest='val_size', type=str2int, default=0, help='# max val crop, set to zero for no crop') 
    parser.add_argument('--fine_size', dest='fine_size', type=str2int, default=128, help='then crop LR to this size')
    parser.add_argument('--disc_size', dest='disc_size', type=str2int, default=128, help='then crop HR to this size during disc')
    parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--ndf', dest='ndf', type=int, default=32, help='# of discri filters in first conv layer')
    parser.add_argument('--ngsrf', dest='ngsrf', type=str2int, default=64, help='# of gen SR filters in first conv layer')
    parser.add_argument('--numResBlocks', dest='numResBlocks', type=str2int, default=16, help='# of resBlocks in SR')
    parser.add_argument('--numResRFBBlocks', dest='numResRFBBlocks', type=str2int, default=1, help='# of resRFBBlocks in SR')
    parser.add_argument('--ndsrf', dest='ndsrf', type=str2int, default=64, help='# of discri SR filters in first conv layer')
    parser.add_argument('--lr', dest='lr', type=float, default=1e-4, help='initial learning rate for adam')
    parser.add_argument('--epoch_step', dest='epoch_step', type=str2int, default=100, help='# of epoch to decay lr')
    parser.add_argument('--lrType', dest='lrType', type=str, default='halfLife', help='halfLife,cosineAnneal,constant,')
    parser.add_argument('--phase', dest='phase', type=str, default='train', help='train, test')
    # augmentation arguments - contrast adjustment?
    
    # Model IO
    parser.add_argument('--save_freq', dest='save_freq', type=str2int, default=10, help='save a model every save_freq epochs')
    parser.add_argument('--print_freq', dest='print_freq', type=str2int, default=10, help='print the validation images every X epochs')
    parser.add_argument('--continue_train', dest='continue_train', type=str2bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
    parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoints', help='models are saved here')
    parser.add_argument('--modelName', dest='modelName', default='sigmaNetTest', help='models are loaded here')
    
    # architecture switches
    # it would be nice to have: p2p(GAN), C1GAN, SR(GAN), SigmaNet - for now, only add switches if necessary
    parser.add_argument('--srFlag', dest='srFlag', type=str2bool, default=True, help='srFlag')
    parser.add_argument('--sigmaType', dest='sigmaType', type=str, default='omega', help='sigma, delta, gamma, omega') # add gamma and omega mode later
    parser.add_argument('--ganFlag', dest='ganFlag', type=str2bool, default=False, help='ganFlag')
    
    # model switches
    # ResNet-like, U-Net, EDSR, SRGAN-D, PatchGAN-D, EfficientNet, RFB-RRDB
    parser.add_argument('--generatorType', dest='generatorType', type=str, default='edsr', help='edsr,rrdb,rrfdb-rrdb')
    # loss switches    
    # L1, L2, scgan, lsgan, relscgan, coefficients
    parser.add_argument('--sigmaCouplingFlag', dest='sigmaCouplingFlag', type=str2bool, default=True, help='sigmaCouplingFlag')
    parser.add_argument('--cyclePixelwiseLoss', dest='cyclePixelwiseLoss', type=str, default='L1', help='L1, L2')
    parser.add_argument('--cycleDiscLoss', dest='cycleDiscLoss', type=str, default='LS', help='LS, SC, RelSC, RelLS')
    parser.add_argument('--srPixelwiseLoss', dest='srPixelwiseLoss', type=str, default='L1', help='L1, L2')
    parser.add_argument('--srDiscLoss', dest='srDiscLoss', type=str, default='SC', help='SC, RelSC')
    # loss coefficients 
    parser.add_argument('--cycleAdv_lambda', dest='cycleAdv_lambda', type=str2float, default=0.1, help='weight on Adv term for normal cycle')
    parser.add_argument('--srAdv_lambda', dest='srAdv_lambda', type=str2float, default=1e-3, help='weight on Adv term for normal sr')
    parser.add_argument('--sigmaCoupling_lambda', dest='sigmaCoupling_lambda', type=str2float, default=0.1, help='sigmaCouplingFlag')
#    parser.add_argument('--idt_lambda', dest='idt_lambda', type=str2float, default=0.0, help='weight assigned to the a2b identity loss function') # b2b should give b
#    parser.add_argument('--tv_lambda', dest='tv_lambda', type=str2float, default=0.0, help='weight assigned to the a2b total variation loss function')
#    parser.add_argument('--L1_sr_lambda', dest='L1_sr_lambda', type=str2float, default=10.0, help='weight on L1 term in the SR cycle') # low since patchGAN doesnt have dense summation?
#    parser.add_argument('--glcm_sr_lambda', dest='glcm_sr_lambda', type=str2float, default=0.0, help='weight on glcm term in the SR cycle')
#    parser.add_argument('--idt_sr_lambda', dest='idt_sr_lambda', type=str2float, default=0.0, help='weight assigned to the SR identity loss function')
#    parser.add_argument('--tv_sr_lambda', dest='tv_sr_lambda', type=str2float, default=0.0, help='weight assigned to the SR total variation loss function') # this is a crutch. avoid it. if needed, tune it carefuly. div2k accepts 1-2e-4, amd fails at 1e-3 vs 10


    # testing arguments
    parser.add_argument('--testInputs', dest='testInputs', default='./testFolder/dmaxSamples/', help='test input images are here')
    parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
    parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
    args = parser.parse_args()

    gpuList=args.gpuIDs
    args.numGPUs = len(gpuList.split(','))
    if args.numGPUs<=4:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
        os.environ["CUDA_VISIBLE_DEVICES"]=gpuList
    return args
