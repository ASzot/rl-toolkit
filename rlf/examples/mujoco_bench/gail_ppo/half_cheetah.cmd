python -m rlf.examples.train --alg gail_ppo --prefix halfcheetah_gail_ppo --env-name HalfCheetah-v3 --save-interval 100000000 --num-env-steps 5e6 --eval-interval 1000000 --num-eval 5 --eval-num-processes 1 --traj-load-path ./data/traj/HalfCheetah-v3/319-HC-31-KZ-walker_ppo/trajs.pt --gail-reward-norm True --lr 3e-4  --normalize-env False --reward-type gail --log-interval 10
