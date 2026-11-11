#!/bin/bash
# after A5 r1 (PID 438049): A5b (T8, 150 turns). A5 r2 (60 turns) dropped: 3/3 r1 episodes ended at the 60-turn cap
# with no submission (decision logged before A5b ran).
cd /home/ec2-user/edit_hunt
while kill -0 438049 2>/dev/null; do sleep 30; done
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python -m edithunt.env.suite_run --model gpt-6.1-sol --tasks T8_undo --n 4 --blackbox_tasks "" --max_turns 150 --max_cost 0.90 --effort medium --run_tag b150 > logs/t2v21/agent_A5b.log 2>&1
