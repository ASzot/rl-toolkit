# Pretending like num-cpu=16 and num-envs=2 and batch-size 128
# num-steps needs to be set to the episode length
python tests/dev/her/def.py --prefix 'her-test' --num-env-steps 5e5 --env-name "FetchReach-v1" --eval-interval -1 --log-smooth-len 10 --save-interval -1 --lr 0.001 --critic-lr 0.001 --tau 0.05 --warmup-steps 0 --update-every 1 --trans-buffer-size 1000000 --batch-size 128 --linear-lr-decay False --max-grad-norm -1 --noise-std 0.25 --noise-type gaussian --num-processes 32 --num-steps 100 --updates-per-batch 40
