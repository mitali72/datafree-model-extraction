cd dfme;

python3 train.py --dataset 400 --ckpt checkpoint/teacher/cifar10-resnet34_8x.pt --device 0 --grad_m 1 --query_budget 20 --log_dir save_results/cifar10  --lr_G 1e-4 --student_model resnet50 --loss l1 --nz 559 --batch_size 1 ;
