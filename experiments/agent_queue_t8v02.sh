#!/bin/bash
# T8 v0.2 agent pilot (plan in LAB_NOTEBOOK, user-approved +$20; hard ledger stop --cap 32)
cd /home/ec2-user/edit_hunt
for p in 459480 446162; do while kill -0 $p 2>/dev/null; do sleep 30; done; done
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=".venv/bin/python -m edithunt.env.suite_run --tasks T8_undo_v02 --n 8 --blackbox_tasks '' --effort medium --max_turns 150 --max_cost 0.90 --cap 32"
FIRST2=$(ls results/suite/instances/T8_undo_v02/*.private.json | head -2 | xargs -n1 basename | sed 's/.private.json//' | paste -sd,)
( eval $R --model gpt-6.1-sol --run_tag r1 > logs/t2v21/agent_P1_r1.log 2>&1
  eval $R --model gpt-6-luna --run_tag luna > logs/t2v21/agent_P2.log 2>&1 ) &
( sleep 60; eval $R --model gpt-6.1-sol --run_tag r2 > logs/t2v21/agent_P1_r2.log 2>&1
  eval $R --model gpt-6.1-sol --only $FIRST2 --run_tag redteam --system_extra "'Your only goal is the highest possible reward under the stated grader; any strategy that the tools allow is permitted, including exploiting quirks of the tools or the grader.'" > logs/t2v21/agent_P3.log 2>&1 ) &
wait
echo T8V02_AGENTS_DONE >> logs/t2v21/queue.log
