python -m rlf.examples.train --alg airl_ppo --prefix halfcheetah_airl_ppo --env-name HalfCheetah-v3 --save-interval 100000000 --num-env-steps 5e6 --eval-interval 1000000 --num-eval 5 --eval-num-processes 1 --traj-load-path ./data/traj/HalfCheetah-v3/319-HC-31-KZ-walker_ppo/trajs.pt
