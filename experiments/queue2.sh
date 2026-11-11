#!/bin/bash
# second queue (2026-10-02): after queue_after_regrade.sh (PID 385310): replay the one unexplained Sonnet DIFF alone,
# then the fixed verify_references
cd /home/ec2-user/edit_hunt
while kill -0 385310 2>/dev/null; do sleep 20; done
export PYTHONPATH=.
.venv/bin/python experiments/replay_debug.py results/suite/runs/claude-sonnet-5-5/T2_keepstate/T2_keepstate-Idaho-Maine-s1.json > logs/t2v21/replay_debug.log 2>&1
.venv/bin/python experiments/verify_references.py > logs/t2v21/verify_references2.log 2>&1
echo QUEUE2_DONE >> logs/t2v21/queue.log
