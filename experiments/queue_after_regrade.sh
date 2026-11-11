#!/bin/bash
# Sequential GPU queue (2026-10-02): waits for the regrade shards by literal PID, then regression checks and H2/H4.
cd /home/ec2-user/edit_hunt
for p in 379373 379374; do while kill -0 $p 2>/dev/null; do sleep 20; done; done
export PYTHONPATH=.
.venv/bin/python experiments/verify_references.py > logs/t2v21/verify_references.log 2>&1
.venv/bin/python experiments/verify_review_fixes.py > logs/t2v21/verify_replay.log 2>&1
.venv/bin/python experiments/h2_readouts.py > logs/t2v21/h2_readouts.log 2>&1
.venv/bin/python experiments/h4_neighbors.py > logs/t2v21/h4_neighbors.log 2>&1
echo QUEUE_DONE >> logs/t2v21/queue.log
