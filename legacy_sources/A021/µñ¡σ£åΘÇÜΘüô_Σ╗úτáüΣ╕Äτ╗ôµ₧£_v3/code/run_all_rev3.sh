for c in SOD LO DG BLAST SV SB; do python tcx_run.py $c > logs_$c.txt 2>&1; done
echo ALLDONE > alldone3.txt
