python -m rlf.examples.train --alg sac --prefix hopper_sac --env-name Hopper-v3 --save-interval 100000000 --num-env-steps 5e6 --eval-interval 10000 --num-render 0 --num-eval 10 --eval-num-processes 1 --trans-buffer-size 1e6 --lr 1e-4 --init-temperature 0.1 --n-rnd-steps 0 --critic-target-update-freq 2 --batch-size 1024 --normalize-env False  --learnable-temp True --dist-q-hidden-dim 1024 --policy-hidden-dim 1024  --use-proper-time-limits True