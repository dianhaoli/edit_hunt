#!/bin/bash
# after the T8 v0.2 agent queue (PID 461157): fill in crashed/missing P1 episodes (suite_run skips existing records),
# then resume the T2 v2.1 budget curve (H3)
cd /home/ec2-user/edit_hunt
while kill -0 461157 2>/dev/null; do sleep 30; done
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=".venv/bin/python -m edithunt.env.suite_run --tasks T8_undo_v02 --n 8 --blackbox_tasks '' --effort medium --max_turns 150 --max_cost 0.90 --cap 32"
eval $R --model gpt-6.1-sol --run_tag r1 >> logs/t2v21/agent_P1_r1.log 2>&1
eval $R --model gpt-6.1-sol --run_tag r2 >> logs/t2v21/agent_P1_r2.log 2>&1
echo FILLIN_DONE >> logs/t2v21/queue.log
.venv/bin/python experiments/t2v21_library.py --methods opt_keep20,opt_keep10_lr02,opt_keep20_lr02,opt_keep40 >> logs/t2v21/library.0.log 2>&1
echo BUDGET_DONE >> logs/t2v21/queue.log
