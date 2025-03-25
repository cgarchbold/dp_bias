#Test
#python train.py --exp_name test --epochs 1 --lr 0.001 --batch_size 32 --private --epsilon 10 --delta 0.001 #--pretrained --freeze_backbone 

#Baseline Resnet18 no privacy, no bias
python train.py --exp_name resnet18_nopriv_nobias --epochs 25 --lr 0.001 --batch_size 32 --pretrained

#Baseline Resnet18 no privacy, yes bias
python train.py --exp_name resnet18_nopriv --epochs 25 --lr 0.001 --batch_size 32 --pretrained --apply_bias

#Resnet18 DPeps=1, no bias
python train.py --exp_name resnet18_eps1_nobias --epochs 25 --lr 0.001 --batch_size 32 --private --epsilon 1 --delta 0.001
#Resnet18 DPeps=10, no bias
python train.py --exp_name resnet18_eps10_nobias --epochs 25 --lr 0.001 --batch_size 32 --private --epsilon 10 --delta 0.001

#Resnet18 DPeps=1, yes bias
python train.py --exp_name resnet18_eps1 --epochs 25 --lr 0.001 --batch_size 32 --private --epsilon 1 --delta 0.001 --apply_bias
#Resnet18 DPeps=10, yes bias
python train.py --exp_name resnet18_eps10 --epochs 25 --lr 0.001 --batch_size 32 --private --epsilon 10 --delta 0.001 --apply_bias

#----------------------------------------------------------------------------------------
#Resnet18 DPeps=1, no bias, freeze strategy
python train.py --exp_name resnet18_eps1_nobias_freeze --epochs 25 --lr 0.0001 --batch_size 32 --private --epsilon 1 --delta 0.001 --pretrained --freeze_backbone
#Resnet18 DPeps=10, no bias, freeze strategy
python train.py --exp_name resnet18_eps10_nobias_freeze --epochs 25 --lr 0.0001 --batch_size 32 --private --epsilon 10 --delta 0.001 --pretrained --freeze_backbone

#Resnet18 DPeps=1, yes bias, freeze strategy
python train.py --exp_name resnet18_eps1_freeze --epochs 25 --lr 0.0001 --batch_size 32 --private --epsilon 1 --delta 0.001 --apply_bias --pretrained --freeze_backbone
#Resnet18 DPeps=10, yes bias, freeze strategy
python train.py --exp_name resnet18_eps10_freeze --epochs 25 --lr 0.0001 --batch_size 32 --private --epsilon 10 --delta 0.001 --apply_bias --pretrained --freeze_backbone

