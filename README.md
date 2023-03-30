# How to run

main.py --net=EEGNet --label=0,1,2,3 --gpu=0 --sch=exp --gamma=0.999 --epoch=50 -lr=2e-4 --stamp=baseline --seed=42 --batch_size=72 -wd=2e-4 --train_subject=9


# Arguments you must change
net: backbone network (if you want to use other networks)

gpu: index of gpu you'd like to use

sch: scheduler (specify gamma value when you use exp)

epoch: the number of epochs

lr: learning rate

stamp: 'stamp' will be the save path of the experimental result

seed: random seed

batch_size: batch size

wd: weight decay

train_subject: subject you'd like to train & test


# Notice

실험 후 ./result/{stamp}/에 실험자 별 폴더가 생성됩니다.

After running "main.py", directories for each subject will be created in "./result/{stamp}/".

각 폴더 안에는 args.json, checkpoint (DIR), log_dict.json 파일 및 디렉터리가 있는데
"args.json, checkpoint, log_dict.json" will be created in the path.

args.json은 실험에 사용된 args가 저장된 파일

args.json contains the information of arguments for the experiment.

log_dict.json은 실험 결과 (train loss, train acc, test/val loss, test/val acc) 가 저장됩니다.

The experimental results including train loss/acc, test(val) loss/acc will be saved in "log_dict.json"

checkpoint에는 모델 파라미터가 저장됩니다. 50에폭마다 저장되게 설정되어 있으며 이는 코드 내에서 수정가능합니다. 

The parameters of trained model will be saved in "checkpoint". 