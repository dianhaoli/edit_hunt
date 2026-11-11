#!/bin/bash
cd /home/ec2-user/edit_hunt
while kill -0 456455 2>/dev/null; do sleep 15; done
PYTHONPATH=. .venv/bin/python experiments/t2v21_failure_audit.py > logs/t2v21/t2v21_failure_audit.log 2>&1
echo AUDIT_DONE >> logs/t2v21/queue.log
