for i in {713178..724607} # 180V
do
    python3 merge_scope_etroc.py --run_number $i
done
