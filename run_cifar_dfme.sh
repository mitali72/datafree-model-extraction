cd dfme;

python3 train.py --num_classes 600 --model_id "blackbox" --ckpt checkpoint/teacher/cifar10-resnet34_8x.pt --grad_m 1 --query_budget 20 --log_dir save_results/cifar10  --lr_G 1e-4 --student_model stam --loss l1 --nz 559 --batch_size 1 ;
