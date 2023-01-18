for RUN in 1 2 3 4 5 6 7 8 ; do 
  for METHOD in "" "CROP Action" "CROP Object" "CROP Radius"; do
    O="results/out-eval/PPO/$METHOD"; mkdir -p "$O"
    nohup echo "Running PPO $METHOD $RUN" &> "$O/$RUN.out" &
    nohup python -m run PPO $METHOD --stop --path results/1-evaluation &> "$O/$RUN.out" &
    sleep 1s
  done
  sleep 1h
done
