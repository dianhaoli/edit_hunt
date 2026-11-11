#!/bin/bash
# third queue (2026-10-02): after queue2 (PID 419007): T8 pilot set, then resume the T2 v2.1 library (budget curve)
cd /home/ec2-user/edit_hunt
while kill -0 419007 2>/dev/null; do sleep 20; done
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python experiments/t8_pilot_set.py > logs/t2v21/t8_pilot_set.log 2>&1
.venv/bin/python experiments/t2v21_library.py --methods opt_keep20,opt_keep10_lr02,opt_keep20_lr02,opt_keep40,opt_keep100_s1,opt_keep100_more,ungated_keep100,opt_keep150,opt_keep100_s2 >> logs/t2v21/library.0.log 2>&1
echo QUEUE3_DONE >> logs/t2v21/queue.log
