python -m rlf.examples.train --alg sac --prefix ant_sac --env-name Ant-v3 --save-interval 100000000 --num-env-steps 5e6 --eval-interval 1000000 --num-eval 5 --eval-num-processes 1 --trans-buffer-size 1e6 --lr 1e-3 --init-temperature 0.2 --n-rnd-steps 10000 --critic-target-update-freq 1 --sac-update-freq 50 --sac-update-epochs 50 --batch-size 128
