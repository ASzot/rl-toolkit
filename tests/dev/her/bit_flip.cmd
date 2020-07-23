# num-steps = bit-flip-n 
# num-env-steps = 1000 * 16 * bit-flip-n
python tests/dev/her/def.py --prefix 'her-test' --env-name "BitFlip-v0" --log-smooth-len 10 --save-interval -1 --lr 0.001 --trans-buffer-size 1000000 --batch-size 32 --linear-lr-decay False --max-grad-norm -1 --num-processes 16 --updates-per-batch 40 --log-interval 5 --num-render 0 --eval-interval -1 --eps-start 0.2 --eps-end 0.02 --eps-decay 45 --gamma 0.98 --normalize-env False --num-env-steps 160000 --num-steps 10 --bit-flip-n 10 
